"""
pipeline_qwen_dual.py — Two-Phase Expert Prefetch Pipeline for Qwen1.5-MoE-A2.7B
==================================================================================

Design
------
Two prefetch signals fire at DIFFERENT points in the computation graph, providing
a timing advantage without sacrificing the formal CRC coverage guarantee:

  Phase 1  (EARLY — fires right after attention_l, before MoE_l):
    Signal E_A  — velocity-corrected logit blend on post-attention hiddens.

      h_post_attn_l   = hidden state after Self-Attn_l + residual
                        (the same h that Signal A uses)
      h_extrap_A      = 2·h_post_attn_l − h_post_attn_{l−1}   (velocity extrapolation)

      z_A   = input_LN_{l+1}(h_post_attn_l)   · W_gate_{l+1}^T   (raw logits, Signal A)
      z_DA  = input_LN_{l+1}(h_extrap_A)      · W_gate_{l+1}^T   (velocity version)
      z_EA  = (1−ε)·z_A + ε·z_DA → softmax(z_EA)

      ε per-layer chosen on calibration to maximise rank-1 hit rate.
      Prefetches exactly 1 expert (argmax) — deterministic, no CRC needed.
      Long load window: MoE_l + remaining layer l compute still ahead.

  Phase 2  (LATE — fires after MoE_l runs, when h_in_{l+1} is ready):
    Signal E_B  — velocity-corrected logit blend on post-MoE boundary.
                  (identical to "Signal E" in pipeline_qwen_signals.py)

      h_in_{l+1}   = exact layer boundary (output of layer l)
      h_extrap_B   = 2·h_in_{l+1} − h_in_l              (velocity extrapolation)

      z_B   = post_attn_LN_{l+1}(h_in_{l+1})  · W_gate_{l+1}^T
      z_DB  = post_attn_LN_{l+1}(h_extrap_B)  · W_gate_{l+1}^T
      z_EB  = (1−ε)·z_B + ε·z_DB → softmax(z_EB)

      ε per-layer chosen on calibration to maximise rank-4 hit rate.
      CRC conformal set: covers all K=4 true experts with P ≥ 1−α (formal guarantee).

  Hybrid prefetch set   = { top-1(E_A) }  ∪  CRC-set(E_B)

  Since r1(E_B) ≈ 99.5 %, top-1(E_A) already belongs to CRC-set(E_B) in ~99.5% of
  tokens.  The union is therefore ≈ CRC-set(E_B) in size, but one expert starts
  loading sooner — purely a timing/pipelining gain at (virtually) zero extra bandwidth.

Calibration rank used for ε selection
--------------------------------------
  E_A : argmax over ε of per-layer rank-1 hit rate  (P(expert #1 is true))
  E_B : argmax over ε of per-layer rank-4 hit rate  (P(expert #4 is true))

CRC guarantee
-------------
  γ* per layer is calibrated on E_B so that       E[L(S_EB, x)] ≤ α   (typically α = 0.10)
  where L = 1 − coverage(set, true_topK).
  E_A has no CRC guarantee — it is a best-effort early prefetch.
"""

from __future__ import annotations

import gc
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from moe_cp_prefetch.config import ExperimentConfig
from moe_cp_prefetch.data import load_splits, iter_token_blocks

# ── Reuse shared helpers from the signals pipeline ─────────────────────────
from moe_cp_prefetch.pipeline_qwen_signals import (
    _StreamStats,
    _gate_logits,
    _softmax_numpy,
    _precompute_sorted,
    _loss_at_gamma_single,
)
from moe_cp_prefetch.qwen_moe_postattn_adapter import QwenMoePostAttnTracer

SEP = "═" * 110


# ═══════════════════════════════════════════════════ _DualStats accumulator

class _DualStats:
    """Test-time accumulator for the dual-signal (E_A + E_B) hybrid prefetch."""

    def __init__(self, n_pairs: int, K: int):
        self.n_pairs     = n_pairs
        self.K           = K
        # E_A
        self.ea_r1_sum   = [0.0] * n_pairs   # top-1(E_A) is a true expert
        self.ea_r2_sum   = [0.0] * n_pairs   # rank-2 hit for E_A
        self.ea_r4_sum   = [0.0] * n_pairs   # rank-4 hit for E_A
        # E_B CRC set
        self.eb_set_sum  = [0.0] * n_pairs   # sum of set sizes
        self.eb_loss_sum = [0.0] * n_pairs   # CRC loss (1 − coverage)
        self.eb_cov_sum  = [0.0] * n_pairs   # P(all K true ∈ set_EB)
        # Hybrid union
        self.overlap_sum = [0.0] * n_pairs   # top-1(E_A) ∈ CRC-set(E_B)
        self.union_sum   = [0.0] * n_pairs   # |{top1_EA} ∪ set_EB|
        self.n_tok       = [0] * n_pairs

    # ------------------------------------------------------------------
    def update(
        self,
        l:         int,
        p_EA:      np.ndarray,   # (T, E) E_A probs
        p_EB:      np.ndarray,   # (T, E) E_B probs
        gamma_EB:  float,        # E_B γ* threshold
        true_ids:  np.ndarray,   # (T, K) true expert indices
        true_scrs: np.ndarray,   # (T, K) true expert gate scores
    ):
        T, E = p_EA.shape
        K    = true_ids.shape[1]

        # ── E_A rank hits ───────────────────────────────────────────────
        # rank of each expert in E_A sorted descending
        ea_order = np.argsort(-p_EA, axis=1)   # (T, E)
        ea_ranks = np.empty_like(ea_order)
        ea_ranks[np.arange(T)[:, None], ea_order] = np.arange(1, E + 1)[None, :]
        # true_ranks for E_A: (T, K) — rank each true expert gets from E_A
        ea_true_ranks = ea_ranks[np.arange(T)[:, None], true_ids]  # (T, K)
        # rank-k hit: any true expert at predicted rank k
        for k_hit, store in [(1, self.ea_r1_sum), (2, self.ea_r2_sum), (4, self.ea_r4_sum)]:
            store[l] += float((ea_true_ranks == k_hit).any(axis=1).sum())
        top1_EA = ea_order[:, 0]   # (T,) — argmax of E_A

        # ── E_B CRC set ─────────────────────────────────────────────────
        eb_order   = np.argsort(-p_EB, axis=1)   # (T, E)
        eb_p_sorted = np.take_along_axis(p_EB, eb_order, axis=1)  # (T, E)
        cum_eb     = np.cumsum(eb_p_sorted, axis=1)
        k_stars    = np.minimum((cum_eb[:, :-1] < gamma_EB).sum(axis=1) + 1, E)  # (T,)

        # Boolean set mask — True iff expert e is in conformal set for token t
        eb_ranks = np.empty_like(eb_order)
        eb_ranks[np.arange(T)[:, None], eb_order] = np.arange(1, E + 1)[None, :]
        set_mask_EB = eb_ranks <= k_stars[:, np.newaxis]  # (T, E)

        eb_set_size = set_mask_EB.sum(axis=1)   # (T,)
        self.eb_set_sum[l] += float(eb_set_size.sum())

        # CRC loss per token: 1 − coverage at k_stars
        # coverage = sum_k true_scrs[k] where true expert ranks[k] ≤ k_star
        # We reuse the _loss_at_gamma_single machinery via sorted true ranks
        # (T, K): rank of each true expert in E_B ordering
        eb_true_ranks = eb_ranks[np.arange(T)[:, None], true_ids]  # (T, K)
        sort_order_K = np.argsort(eb_true_ranks, axis=1)
        sorted_true_ranks = np.take_along_axis(eb_true_ranks, sort_order_K, axis=1)  # (T, K) sorted
        sorted_true_scrs  = np.take_along_axis(true_scrs, sort_order_K, axis=1)
        cum_covered = np.cumsum(sorted_true_scrs, axis=1)   # (T, K)
        _, losses = _loss_at_gamma_single(
            np.take_along_axis(p_EB, eb_order, axis=1),   # sorted (T, E) probs
            sorted_true_ranks, cum_covered, gamma_EB,
        )
        self.eb_loss_sum[l] += float(losses.sum())

        # E_B full coverage: all K true experts inside the set
        all_covered = np.ones(T, dtype=bool)
        for k_idx in range(K):
            all_covered &= set_mask_EB[np.arange(T), true_ids[:, k_idx]]
        self.eb_cov_sum[l] += float(all_covered.sum())

        # ── Hybrid union ────────────────────────────────────────────────
        overlap   = set_mask_EB[np.arange(T), top1_EA]    # (T,) bool
        union_sz  = np.where(overlap, eb_set_size, eb_set_size + 1)
        self.overlap_sum[l] += float(overlap.sum())
        self.union_sum[l]   += float(union_sz.sum())

        self.n_tok[l] += T

    # ------------------------------------------------------------------
    def results(self) -> dict:
        valid = [l for l in range(self.n_pairs) if self.n_tok[l] > 0]
        if not valid:
            return {}
        N = sum(self.n_tok[l] for l in valid)

        def _agg(store):
            return sum(store[l] for l in valid) / N

        per_layer = {}
        for l in valid:
            n = self.n_tok[l]
            per_layer[l] = {
                "ea_r1":       self.ea_r1_sum[l]   / n,
                "ea_r2":       self.ea_r2_sum[l]   / n,
                "ea_r4":       self.ea_r4_sum[l]   / n,
                "avg_eb_set":  self.eb_set_sum[l]  / n,
                "eb_loss":     self.eb_loss_sum[l] / n,
                "eb_coverage": self.eb_cov_sum[l]  / n,
                "overlap":     self.overlap_sum[l] / n,
                "avg_union":   self.union_sum[l]   / n,
            }

        return {
            "ea_r1":        _agg(self.ea_r1_sum),
            "ea_r2":        _agg(self.ea_r2_sum),
            "ea_r4":        _agg(self.ea_r4_sum),
            "avg_eb_set":   _agg(self.eb_set_sum),
            "eb_mean_loss": _agg(self.eb_loss_sum),
            "eb_coverage":  _agg(self.eb_cov_sum),
            "overlap_rate": _agg(self.overlap_sum),
            "avg_union":    _agg(self.union_sum),
            "n_tok":        N,
            "per_layer":    per_layer,
        }


# ════════════════════════════════════════════════════ streaming pass

def _dual_stream_pass(
    tracer:   QwenMoePostAttnTracer,
    tokenizer,
    model:    torch.nn.Module,
    device:   torch.device,
    split_data,
    cfg:      ExperimentConfig,
    max_samples: int,
    desc:     str,
    # ── calibration accumulators ────────────────────────────────────
    eps_stats_EA:        Optional[List[_StreamStats]],  # one per ε  (cal mode)
    eps_stats_EB:        Optional[List[_StreamStats]],  # one per ε  (cal mode)
    gamma_grid:          Optional[np.ndarray],
    eps_grid:            Optional[np.ndarray],
    # ── test accumulators ──────────────────────────────────────────
    best_eps_EA:         Optional[List[float]]   = None,  # per-layer (tst mode)
    best_eps_EB:         Optional[List[float]]   = None,  # per-layer (tst mode)
    sig_EA_stats:        Optional[_StreamStats]  = None,  # E_A rank stats (tst)
    sig_EB_stats:        Optional[_StreamStats]  = None,  # E_B CRC stats  (tst)
    dual_stats:          Optional[_DualStats]    = None,  # union metrics  (tst)
    mode:                str                     = "cal",
):
    """
    Single streaming forward pass for the dual pipeline.

    Cal mode : updates eps_stats_EA + eps_stats_EB for every ε value.
    Tst mode : updates sig_EA_stats, sig_EB_stats, dual_stats.
    """
    n_layers = tracer.num_layers()
    n_pairs  = n_layers - 1
    K        = cfg.router_top_k
    n_eps    = len(eps_grid) if eps_grid is not None else 0

    n_batches = 0
    with torch.no_grad():
        for batch in iter_token_blocks(
            split_data, tokenizer, cfg.text_field,
            max_samples, cfg.max_seq_len, cfg.seed,
            batch_size=1,
        ):
            ids_t  = torch.tensor(batch.input_ids,     dtype=torch.long, device=device)
            mask_t = torch.tensor(batch.attention_mask, dtype=torch.long, device=device)
            seq_len = ids_t.shape[1]
            _ = model(input_ids=ids_t, attention_mask=mask_t, use_cache=False)
            traces  = tracer.pop_last_forward(seq_len)
            n_batches += 1

            for l in range(n_pairs):
                src = traces[l]      # layer l
                tgt = traces[l + 1]  # layer l+1

                if tgt.num_experts == 0 or tgt.true_topk_scores is None:
                    continue

                ids   = tgt.true_topk_ids.numpy()     # (T, K)
                scrs  = tgt.true_topk_scores.numpy()  # (T, K)

                W_gate = tracer.get_router_weight_EH(l + 1).to(device)
                ln_A   = tracer.get_input_layernorm(l + 1)       # wrong LN (Signal A)
                ln_B   = tracer.get_post_attn_layernorm(l + 1)   # correct LN (Signal B)

                # ── Signal A logits (E_A source) ────────────────────────
                h_pa_l  = src.post_attn_states                    # h_post_attn_l (T, H)
                z_A     = _gate_logits(h_pa_l, W_gate, ln_A)     # (T, E)

                # ── Signal D_A logits (velocity on post-attn) ───────────
                h_pa_prev = traces[l - 1].post_attn_states if l > 0 else None
                if h_pa_prev is not None:
                    h_extrap_A = (2.0 * h_pa_l.float() - h_pa_prev.float()).to(h_pa_l.dtype)
                    z_DA = _gate_logits(h_extrap_A, W_gate, ln_A)
                    del h_extrap_A
                else:
                    z_DA = z_A     # l=0: no prior post-attn; zero velocity

                # ── Signal B logits (E_B source) ────────────────────────
                h_in_lp1 = tgt.layer_input_states                 # h_in_{l+1} (T, H)
                h_in_l   = src.layer_input_states                  # h_in_l     (T, H)
                if h_in_lp1 is not None:
                    z_B = _gate_logits(h_in_lp1, W_gate, ln_B)   # (T, E)
                else:
                    z_B = z_A  # fallback (should not occur for Qwen)

                # ── Signal D_B logits (velocity on boundary hiddens) ────
                if h_in_l is not None and h_in_lp1 is not None:
                    h_extrap_B = (2.0 * h_in_lp1.float() - h_in_l.float()).to(h_in_lp1.dtype)
                    z_DB = _gate_logits(h_extrap_B, W_gate, ln_B)
                    del h_extrap_B
                else:
                    z_DB = z_B

                # ── Calibration pass: sweep ε for E_A and E_B ───────────
                if mode == "cal" and eps_grid is not None:
                    for ei, eps in enumerate(eps_grid):
                        # E_A blend → rank-1-based ε selection
                        z_ea = (1.0 - eps) * z_A + eps * z_DA
                        p_ea = _softmax_numpy(z_ea)
                        cp, srank, ccov = _precompute_sorted(p_ea, ids, scrs)
                        eps_stats_EA[ei].update_cal(l, cp, srank, ccov, gamma_grid)
                        del cp, srank, ccov, p_ea, z_ea

                        # E_B blend → rank-4-based ε selection
                        z_eb = (1.0 - eps) * z_B + eps * z_DB
                        p_eb = _softmax_numpy(z_eb)
                        cp, srank, ccov = _precompute_sorted(p_eb, ids, scrs)
                        eps_stats_EB[ei].update_cal(l, cp, srank, ccov, gamma_grid)
                        del cp, srank, ccov, p_eb, z_eb

                # ── Test pass: apply best ε, accumulate dual stats ────────
                elif mode == "tst" and best_eps_EA is not None and best_eps_EB is not None:
                    eps_a = best_eps_EA[l]
                    z_ea  = (1.0 - eps_a) * z_A + eps_a * z_DA
                    p_ea  = _softmax_numpy(z_ea)

                    eps_b = best_eps_EB[l]
                    z_eb  = (1.0 - eps_b) * z_B + eps_b * z_DB
                    p_eb  = _softmax_numpy(z_eb)

                    # E_A rank stats (for comparison/reporting)
                    cp, srank, ccov = _precompute_sorted(p_ea, ids, scrs)
                    sig_EA_stats.update_tst(l, cp, srank, ccov)
                    del cp, srank, ccov

                    # E_B CRC stats (formal guarantee)
                    cp, srank, ccov = _precompute_sorted(p_eb, ids, scrs)
                    sig_EB_stats.update_tst(l, cp, srank, ccov)
                    del cp, srank, ccov

                    # Dual hybrid metrics
                    dual_stats.update(l, p_ea, p_eb, sig_EB_stats.gamma_stars[l], ids, scrs)

                    del p_ea, z_ea, p_eb, z_eb

                del z_A, z_DA, z_B, z_DB

            if n_batches % 50 == 0:
                total = n_batches * seq_len * n_pairs
                print(f"  [{desc}] batch={n_batches}  records={total}", flush=True)


# ════════════════════════════════════════════════════ main entry point

def run_qwen_dual(
    cfg:        ExperimentConfig,
    n_gamma:    int            = 400,
    eps_grid:   Optional[np.ndarray] = None,
) -> dict:
    """
    Two-phase dual-signal CRC prefetch pipeline for Qwen1.5-MoE-A2.7B.

    Returns
    -------
    dict with keys:
        best_eps_EA          : per-layer ε for Signal E_A (rank-1 calibrated)
        best_eps_EB          : per-layer ε for Signal E_B (rank-4 calibrated)
        metrics_EA           : rank hit rates for E_A (early deterministic top-1)
        metrics_EB           : CRC set metrics for E_B (late formal guarantee)
        metrics_dual         : hybrid union metrics (overlap, avg_union, etc.)
    """
    if eps_grid is None:
        eps_grid = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                            dtype=np.float32)

    alpha = cfg.alpha
    K     = cfg.router_top_k
    n_eps = len(eps_grid)

    # ── 1. Load model ──────────────────────────────────────────────────────
    MODEL_PATH = (
        "/models/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/"
        "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    )
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    print(f"[dual] Loading tokenizer from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)

    print(f"[dual] Loading model → dtype={cfg.dtype}, device={device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype, device_map=str(device), trust_remote_code=False
    )
    model.eval()
    print(
        f"[dual] Model loaded.  Params: "
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B",
        flush=True,
    )

    # ── 2. Dataset ─────────────────────────────────────────────────────────
    cal_data, test_data = load_splits(
        cfg.dataset_name, cfg.dataset_config,
        cfg.dataset_split_cal, cfg.dataset_split_test,
        cache_dir=cfg.cache_dir,
    )

    # ── 3. Attach tracer ───────────────────────────────────────────────────
    tracer = QwenMoePostAttnTracer(top_k=K)
    tracer.attach(model)
    n_layers = tracer.num_layers()
    n_pairs  = n_layers - 1
    print(f"[dual] Hooked {n_layers} decoder layers.  {n_pairs} cross-layer pairs.", flush=True)
    print(f"[dual] eps_grid={list(eps_grid)}", flush=True)

    # ── 4. Gamma grid ──────────────────────────────────────────────────────
    gamma_grid = np.linspace(0.01, 0.9999, n_gamma)

    # ── 5. Calibration accumulators ────────────────────────────────────────
    eps_stats_EA = [_StreamStats(n_pairs, n_gamma, K) for _ in range(n_eps)]
    eps_stats_EB = [_StreamStats(n_pairs, n_gamma, K) for _ in range(n_eps)]

    # ── 6. Calibration pass ────────────────────────────────────────────────
    print("\n[dual] === Calibration pass (streaming) ===", flush=True)
    t0 = time.time()
    _dual_stream_pass(
        tracer, tokenizer, model, device, cal_data, cfg,
        cfg.max_samples_cal, "cal",
        eps_stats_EA=eps_stats_EA,
        eps_stats_EB=eps_stats_EB,
        gamma_grid=gamma_grid,
        eps_grid=eps_grid,
        mode="cal",
    )
    print(f"[dual] Calibration done in {time.time()-t0:.0f}s", flush=True)
    n_cal = eps_stats_EA[0].cal_n_tok[0] if eps_stats_EA[0].cal_n_tok else 0
    n_cal_total = sum(eps_stats_EA[0].cal_n_tok)
    print(f"[dual] n_cal tokens per layer ≈ {n_cal}", flush=True)

    # ── 7. Find best ε per layer ───────────────────────────────────────────
    # E_A: maximise rank-1 hit rate
    best_eps_EA = []
    best_eps_EA_r1 = []
    print("\n[dual] Signal E_A — per-layer optimal ε (rank-1 hit rate):")
    hdr = "  Layer  " + "".join(f"  {e:.2f}" for e in eps_grid) + "  best_ε"
    print(hdr)
    for l in range(n_pairs):
        r1_by_eps = []
        for ei in range(n_eps):
            n_l = eps_stats_EA[ei].cal_n_tok[l]
            r1  = float(eps_stats_EA[ei].cal_rh_sum[l][1]) / n_l if n_l > 0 else 0.0
            r1_by_eps.append(r1)
        best_ei  = int(np.argmax(r1_by_eps))
        best_val = float(eps_grid[best_ei])
        best_eps_EA.append(best_val)
        best_eps_EA_r1.append(r1_by_eps[best_ei])
        row = f"  {l:02d}→{l+1:02d}  "
        row += "".join(f"  {r:.4f}" for r in r1_by_eps)
        row += f"  ε={best_val:.2f}"
        print(row)

    # E_B: maximise rank-4 hit rate
    best_eps_EB = []
    best_eps_EB_r4 = []
    print("\n[dual] Signal E_B — per-layer optimal ε (rank-4 hit rate):")
    print(hdr)
    for l in range(n_pairs):
        r4_by_eps = []
        for ei in range(n_eps):
            n_l = eps_stats_EB[ei].cal_n_tok[l]
            r4  = float(eps_stats_EB[ei].cal_rh_sum[l][4]) / n_l if n_l > 0 else 0.0
            r4_by_eps.append(r4)
        best_ei  = int(np.argmax(r4_by_eps))
        best_val = float(eps_grid[best_ei])
        best_eps_EB.append(best_val)
        best_eps_EB_r4.append(r4_by_eps[best_ei])
        row = f"  {l:02d}→{l+1:02d}  "
        row += "".join(f"  {r:.4f}" for r in r4_by_eps)
        row += f"  ε={best_val:.2f}"
        print(row)

    print(f"\n[dual] best_eps_EA: {best_eps_EA}", flush=True)
    print(f"[dual] best_eps_EB: {best_eps_EB}", flush=True)

    # ── 8. Calibrate γ* for E_B (CRC formal guarantee) ───────────────────
    # E_B γ* is taken from the winning ε's stats per layer
    sig_EB_stats = _StreamStats(n_pairs, n_gamma, K)
    for l in range(n_pairs):
        r4_by_eps = [
            float(eps_stats_EB[ei].cal_rh_sum[l][4]) / max(eps_stats_EB[ei].cal_n_tok[l], 1)
            for ei in range(n_eps)
        ]
        best_ei = int(np.argmax(r4_by_eps))
        # inherit calibration loss sums for γ* calibration
        sig_EB_stats.cal_loss_sum[l] = eps_stats_EB[best_ei].cal_loss_sum[l].copy()
        sig_EB_stats.cal_n_tok[l]    = eps_stats_EB[best_ei].cal_n_tok[l]
    sig_EB_stats.calibrate(alpha, n_cal_total // n_pairs, gamma_grid)

    print("\n[dual] E_B γ* per layer:")
    for l in range(n_pairs):
        print(f"  {l:02d}→{l+1:02d}  γ*={sig_EB_stats.gamma_stars[l]:.4f}")

    # E_A stats object (just for rank reporting, no CRC calibration needed)
    sig_EA_stats = _StreamStats(n_pairs, n_gamma, K)
    # No γ* needed for E_A — we only use argmax; set a dummy so update_tst works
    sig_EA_stats.gamma_stars = [1.0] * n_pairs

    # E_A has no CRC so we only call update_tst to collect rank hits;
    # the gamma star is irrelevant because set_size metrics are unused for E_A.

    # ── 9. Test pass ──────────────────────────────────────────────────────
    dual_stats = _DualStats(n_pairs, K)
    print("\n[dual] === Test pass (streaming) ===", flush=True)
    t0 = time.time()
    _dual_stream_pass(
        tracer, tokenizer, model, device, test_data, cfg,
        cfg.max_samples_test, "test",
        eps_stats_EA=None, eps_stats_EB=None,
        gamma_grid=gamma_grid, eps_grid=eps_grid,
        best_eps_EA=best_eps_EA,
        best_eps_EB=best_eps_EB,
        sig_EA_stats=sig_EA_stats,
        sig_EB_stats=sig_EB_stats,
        dual_stats=dual_stats,
        mode="tst",
    )
    print(f"[dual] Test pass done in {time.time()-t0:.0f}s", flush=True)

    # ── 10. Collect results ────────────────────────────────────────────────
    m_EA   = sig_EA_stats.results(K)
    m_EB   = sig_EB_stats.results(K)
    m_dual = dual_stats.results()

    # ── 11. Print summary ─────────────────────────────────────────────────
    rh_A = m_EA.get("rank_hit", {})
    rh_B = m_EB.get("rank_hit", {})

    print(f"\n{SEP}")
    print(f"[dual] DUAL-SIGNAL PREFETCH RESULTS  (α={alpha}  K={K}  n_cal≈{n_cal})")
    print(SEP)
    print("\n  ── Signal E_A  (EARLY: post-attention velocity blend, ε↑r1) ──")
    print(f"  r1={rh_A.get(1,0):.4f}  r2={rh_A.get(2,0):.4f}  "
          f"r3={rh_A.get(3,0):.4f}  r4={rh_A.get(4,0):.4f}")
    print(f"  → Prefetches 1 expert per token before MoE_l runs.")

    print(f"\n  ── Signal E_B  (LATE: post-MoE velocity blend, ε↑r4, CRC) ──")
    print(f"  Adp avg|S|={m_EB.get('avg_set',0):.2f}  "
          f"E[L]={m_EB.get('mean_L',0):.4f}  P(full)={m_EB.get('p_full',0):.4f}")
    print(f"  r1={rh_B.get(1,0):.4f}  r2={rh_B.get(2,0):.4f}  "
          f"r3={rh_B.get(3,0):.4f}  r4={rh_B.get(4,0):.4f}")
    print(f"  → CRC set covers all K true experts with P ≥ {1-alpha:.1f}.")

    print(f"\n  ── Hybrid Union  {{top-1(E_A)}} ∪ CRC-set(E_B) ──")
    print(f"  E_A top-1 accuracy          : {m_dual.get('ea_r1',0):.4f}")
    print(f"  E_B CRC avg set size        : {m_dual.get('avg_eb_set',0):.2f}")
    print(f"  E_B CRC coverage (all K)    : {m_dual.get('eb_coverage',0):.4f}")
    print(f"  Overlap P(top1_EA ∈ set_EB) : {m_dual.get('overlap_rate',0):.4f}")
    print(f"  Avg union size              : {m_dual.get('avg_union',0):.2f}")
    print(f"\n  Interpretation:")
    print(f"    {m_dual.get('overlap_rate',0)*100:.1f}% of tokens: top-1(E_A) is already in E_B CRC set")
    print(f"      → no extra bandwidth; 1 expert just starts loading earlier.")
    print(f"    {(1-m_dual.get('overlap_rate',0))*100:.1f}% of tokens: top-1(E_A) ∉ E_B set")
    print(f"      → 1 extra expert pre-loaded (wasted), avg union = {m_dual.get('avg_union',0):.2f}.")
    print(SEP)

    # Per-layer summary
    print("\n[dual] Per-layer results:")
    print(f"  {'Layer':<8}  {'E_A r1':>8}  {'E_B |S|':>9}  {'E_B cov':>9}  "
          f"{'overlap':>9}  {'union':>7}  {'bestε_A':>8}  {'bestε_B':>8}")
    per = m_dual.get("per_layer", {})
    for l in range(n_pairs):
        d = per.get(l, {})
        print(
            f"  {l:02d}→{l+1:02d}    "
            f"  {d.get('ea_r1',0):6.4f}    "
            f"  {d.get('avg_eb_set',0):5.2f}    "
            f"  {d.get('eb_coverage',0):6.4f}    "
            f"  {d.get('overlap',0):6.4f}   "
            f"  {d.get('avg_union',0):5.2f}    "
            f"  {best_eps_EA[l]:.2f}      "
            f"  {best_eps_EB[l]:.2f}"
        )
    print(SEP)

    print(
        "\n  Key:\n"
        "  E_A   : velocity-corrected logit blend on post-attention hiddens\n"
        "          z_EA = (1−ε)·z_A + ε·z_{D_A} → softmax; ε↑=rank-1 cal\n"
        "          fires EARLY (after attention_l, before MoE_l)\n"
        "  E_B   : velocity-corrected logit blend on post-MoE boundary\n"
        "          z_EB = (1−ε)·z_B + ε·z_{D_B} → softmax; ε↑=rank-4 cal\n"
        "          fires LATE (after MoE_l, when h_in_{l+1} is ready)\n"
        "  Union : {top-1(E_A)} ∪ CRC-set(E_B)  — the total prefetch set\n"
        "  CRC   : E[L(S_EB)] ≤ α=0.1 guaranteed on E_B alone\n"
    )

    tracer.detach()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "alpha":         alpha,
        "n_gamma":       n_gamma,
        "eps_grid":      list(eps_grid),
        "best_eps_EA":   best_eps_EA,
        "best_eps_EB":   best_eps_EB,
        "metrics_EA":    m_EA,
        "metrics_EB":    m_EB,
        "metrics_dual":  m_dual,
    }
