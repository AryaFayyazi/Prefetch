"""
pipeline_gemma4_dual.py — Two-Phase Expert Prefetch Pipeline for Gemma4-26B-A4B
=================================================================================

Design
------
All 30 Gemma4TextDecoderLayer blocks contain both a shared MLP and a MoE router.
The router receives h_post_attn_l directly (no extra LayerNorm), so:

  Phase 1  (EARLY — fires after layer l-1 completes, before attention at layer l):
    Signal E_A — velocity-corrected router-logit blend using the previous layer's
                 output as a proxy for h_post_attn_l.

      h_in_l        = output of decoder layer l-1         (= h_post_attn_{l-1} + FFN)
      h_extrap_A    = 2·h_in_l − h_in_{l−1}              (velocity extrapolation)

      z_A    = router_logits(h_in_l)                      (Signal A raw logits)
      z_DA   = router_logits(h_extrap_A)                  (velocity version)
      z_EA   = (1−ε)·z_A + ε·z_DA → softmax(z_EA)

      ε chosen per-layer on calibration to maximise rank-1 hit rate.
      Prefetch: top-1(p_EA) — 1 expert, long load window.

  Phase 2  (LATE — fires when h_post_attn_l is computed, before shared MLP finishes):
    Signal E_B — velocity-corrected blend using the router's exact input.

      h_post_attn_l   = hidden state after Self-Attn_l + residual
                        (the EXACT input the router of layer l receives)
      h_extrap_B      = 2·h_post_attn_l − h_post_attn_{l−1}

      z_B    = router_logits(h_post_attn_l)               (Signal B raw logits)
      z_DB   = router_logits(h_extrap_B)
      z_EB   = (1−ε)·z_B + ε·z_DB → softmax(z_EB)

      ε chosen per-layer on calibration to maximise rank-8 hit rate (K=8).
      CRC conformal set: covers all K=8 true experts with E[L] ≤ α.

  Hybrid prefetch set = { top-1(E_A) } ∪ CRC-set(E_B)

Model constants — Gemma4-26B-A4B (gemma4_text config):
  hidden_size   H = 2816   num_experts  E = 128   top_k_experts  K = 8
  num_layers    L = 30     ALL 30 layers have MoE
"""

from __future__ import annotations

import gc
import os
import time
from typing import List, Optional

import numpy as np
import torch

from moe_cp_prefetch.data import load_splits, iter_token_blocks
from moe_cp_prefetch.gemma4_adapter import (
    MODEL_PATH,
    Gemma4DualTracer,
    Gemma4RouterParams,
    extract_router_params,
    router_logits_np,
)

# Reuse generic helpers from the signals pipeline (model-agnostic numpy utils)
from moe_cp_prefetch.pipeline_qwen_signals import (
    _StreamStats,
    _softmax_numpy,
    _precompute_sorted,
    _loss_at_gamma_single,
)

SEP = "═" * 110


# ════════════════════════════════════════════════════ _DualStats accumulator

class _DualStats:
    """Test-time accumulator for dual-signal (E_A + E_B) hybrid prefetch."""

    def __init__(self, n_layers: int, K: int):
        self.n_layers    = n_layers
        self.K           = K
        # E_A per-layer counts
        self.ea_r1_sum   = [0.0] * n_layers   # rank-1 hit (top-1 is a true expert)
        self.ea_r2_sum   = [0.0] * n_layers
        self.ea_r4_sum   = [0.0] * n_layers
        self.ea_r8_sum   = [0.0] * n_layers
        # E_B CRC set
        self.eb_set_sum  = [0.0] * n_layers   # sum of set sizes
        self.eb_loss_sum = [0.0] * n_layers   # CRC loss
        self.eb_cov_sum  = [0.0] * n_layers   # P(all K true ∈ set_EB)
        # Hybrid union
        self.overlap_sum = [0.0] * n_layers   # P(top-1(E_A) ∈ CRC-set(E_B))
        self.union_sum   = [0.0] * n_layers   # |{top1_EA} ∪ set_EB|
        self.n_tok       = [0] * n_layers

    # ------------------------------------------------------------------
    def update(
        self,
        l:         int,
        p_EA:      np.ndarray,   # (T, E)
        p_EB:      np.ndarray,   # (T, E)
        gamma_EB:  float,
        true_ids:  np.ndarray,   # (T, K)
        true_scrs: np.ndarray,   # (T, K)
    ):
        T, E = p_EA.shape
        K    = true_ids.shape[1]

        # E_A rank assignments
        ea_order = np.argsort(-p_EA, axis=1)          # (T, E) desc
        ea_ranks = np.empty_like(ea_order)
        ea_ranks[np.arange(T)[:, None], ea_order] = np.arange(1, E + 1)[None, :]
        ea_true_ranks = ea_ranks[np.arange(T)[:, None], true_ids]  # (T, K)

        for k_hit, store in [
            (1, self.ea_r1_sum),
            (2, self.ea_r2_sum),
            (4, self.ea_r4_sum),
            (8, self.ea_r8_sum),
        ]:
            store[l] += float((ea_true_ranks == k_hit).any(axis=1).sum())
        top1_EA = ea_order[:, 0]   # (T,)

        # E_B CRC set
        eb_order    = np.argsort(-p_EB, axis=1)
        eb_p_sorted = np.take_along_axis(p_EB, eb_order, axis=1)  # (T, E)
        cum_eb      = np.cumsum(eb_p_sorted, axis=1)
        k_stars     = np.minimum((cum_eb[:, :-1] < gamma_EB).sum(axis=1) + 1, E)  # (T,)

        eb_ranks = np.empty_like(eb_order)
        eb_ranks[np.arange(T)[:, None], eb_order] = np.arange(1, E + 1)[None, :]
        set_mask_EB = eb_ranks <= k_stars[:, np.newaxis]  # (T, E)

        eb_set_size = set_mask_EB.sum(axis=1)
        self.eb_set_sum[l] += float(eb_set_size.sum())

        # CRC loss
        eb_true_ranks = eb_ranks[np.arange(T)[:, None], true_ids]  # (T, K)
        sort_k = np.argsort(eb_true_ranks, axis=1)
        sorted_true_ranks = np.take_along_axis(eb_true_ranks, sort_k, axis=1)
        sorted_true_scrs  = np.take_along_axis(true_scrs, sort_k, axis=1)
        cum_covered = np.cumsum(sorted_true_scrs, axis=1)
        _, losses = _loss_at_gamma_single(
            eb_p_sorted, sorted_true_ranks, cum_covered, gamma_EB,
        )
        self.eb_loss_sum[l] += float(losses.sum())

        # Full coverage
        all_covered = np.ones(T, dtype=bool)
        for k_idx in range(K):
            all_covered &= set_mask_EB[np.arange(T), true_ids[:, k_idx]]
        self.eb_cov_sum[l] += float(all_covered.sum())

        # Hybrid union
        overlap  = set_mask_EB[np.arange(T), top1_EA]   # (T,) bool
        union_sz = np.where(overlap, eb_set_size, eb_set_size + 1)
        self.overlap_sum[l] += float(overlap.sum())
        self.union_sum[l]   += float(union_sz.sum())

        self.n_tok[l] += T

    # ------------------------------------------------------------------
    def results(self) -> dict:
        valid = [l for l in range(self.n_layers) if self.n_tok[l] > 0]
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
                "ea_r8":       self.ea_r8_sum[l]   / n,
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
            "ea_r8":        _agg(self.ea_r8_sum),
            "avg_eb_set":   _agg(self.eb_set_sum),
            "eb_mean_loss": _agg(self.eb_loss_sum),
            "eb_coverage":  _agg(self.eb_cov_sum),
            "overlap_rate": _agg(self.overlap_sum),
            "avg_union":    _agg(self.union_sum),
            "n_tok":        N,
            "per_layer":    per_layer,
        }


# ════════════════════════════════════════════════════ streaming pass

def _gemma4_stream_pass(
    tracer:       Gemma4DualTracer,
    router_params: List[Optional[Gemma4RouterParams]],
    tokenizer,
    model:        torch.nn.Module,
    split_data,
    max_samples:  int,
    max_seq_len:  int,
    text_field:   str,
    seed:         int,
    desc:         str,
    # ── calibration accumulators (mode="cal") ──────────────────────────────
    eps_stats_EA: Optional[List[_StreamStats]],   # [n_eps], one per ε
    eps_stats_EB: Optional[List[_StreamStats]],   # [n_eps], one per ε
    gamma_grid:   Optional[np.ndarray],
    eps_grid:     Optional[np.ndarray],
    # ── test accumulators (mode="tst") ─────────────────────────────────────
    best_eps_EA:  Optional[List[float]]  = None,  # per-layer best ε for E_A
    best_eps_EB:  Optional[List[float]]  = None,  # per-layer best ε for E_B
    sig_EA_stats: Optional[_StreamStats] = None,  # E_A rank metrics
    sig_EB_stats: Optional[_StreamStats] = None,  # E_B CRC metrics
    dual_stats:   Optional[_DualStats]   = None,  # hybrid union metrics
    mode:         str                    = "cal",
):
    """
    One streaming forward pass that accumulates either calibration or test stats.

    For each sequence → run forward (captures traces via hooks) → for each
    MoE layer l compute z_A, z_DA, z_B, z_DB in numpy → sweep ε (cal) or
    apply best ε (tst) → update stat accumulators → clear traces.

    Works with device_map="auto" (multi-GPU); tensor operations are pure numpy
    after the forward pass.
    """
    n_layers = tracer.n_layers   # 30 for Gemma4-26B-A4B
    n_eps    = len(eps_grid) if eps_grid is not None else 0
    n_batches = 0
    # Cross-sequence carry for layer 0 Signal A:
    # After each sequence we store h_in_list[n_layers-1][-1:] (last token,
    # shape (1, H)).  At l=0 of the *next* sequence we broadcast it to (T, H)
    # so E_A can predict layer-0 routing from the previous sequence's final
    # residual stream — exactly how an online prefetcher would behave.
    carry_h: Optional[np.ndarray] = None   # (1, H) float32, updated each seq

    with torch.no_grad():
        for batch in iter_token_blocks(
            split_data, tokenizer, text_field,
            max_samples, max_seq_len, seed,
            batch_size=1,
        ):
            ids_t  = torch.tensor(batch.input_ids,      dtype=torch.long)
            mask_t = torch.tensor(batch.attention_mask, dtype=torch.long)

            # Forward — hooks populate tracer in-place
            # No need to specify device; device_map="auto" handles placement
            _ = model(input_ids=ids_t, attention_mask=mask_t, use_cache=False)

            n_batches += 1
            seq_len = ids_t.shape[1]

            # ── per-layer signal computation ──────────────────────────────
            for l in range(n_layers):
                rp = router_params[l]
                if rp is None:
                    continue                          # layer has no MoE router

                ids  = tracer.true_topk_ids[l]       # (T, K) or None
                scrs = tracer.true_topk_scrs[l]      # (T, K) or None
                if ids is None or scrs is None:
                    continue

                h_post = tracer.h_post_attn_list[l]  # (T, H) — E_B source
                if h_post is None:
                    continue

                T = h_post.shape[0]

                # ── Signal B logits (exact router input) ──────────────────
                z_B  = router_logits_np(h_post, rp)   # (T, E)

                # ── Signal D_B logits (velocity on h_post_attn) ───────────
                h_post_prev = tracer.h_post_attn_list[l - 1] if l > 0 else None
                if h_post_prev is not None and h_post_prev.shape[0] == T:
                    h_extra_B = (2.0 * h_post - h_post_prev).astype(np.float32)
                    z_DB = router_logits_np(h_extra_B, rp)
                else:
                    z_DB = z_B                        # no velocity; fallback

                # ── Signal A logits (previous layer output as proxy) ──────
                #    h_in_l  = output of layer l-1  (available for l >= 1)
                #    l == 0  → use cross-sequence carry (last layer last token
                #              of the previous sequence), broadcast to (T, H).
                h_in_l   = tracer.h_in_list[l - 1] if l > 0 else None
                h_in_lm1 = tracer.h_in_list[l - 2] if l > 1 else None

                # Layer-0 carry: broadcast previous sequence's final residual
                if h_in_l is None and carry_h is not None:
                    # carry_h shape (1, H) → tile to (T, H)
                    h_in_l = np.ascontiguousarray(
                        np.broadcast_to(carry_h, (T, carry_h.shape[1]))
                    ).astype(np.float32)
                    # h_in_lm1 stays None → no velocity at l=0

                has_ea = h_in_l is not None

                if has_ea:
                    h_in_l = h_in_l.astype(np.float32)
                    z_A = router_logits_np(h_in_l, rp)   # (T, E)
                    if h_in_lm1 is not None and h_in_lm1.shape[0] == T:
                        h_extra_A = (2.0 * h_in_l - h_in_lm1.astype(np.float32))
                        z_DA = router_logits_np(h_extra_A, rp)
                    else:
                        z_DA = z_A                    # no velocity; fallback
                else:
                    z_A = z_B; z_DA = z_DB           # first seq l=0: use B

                # ── Calibration: sweep all ε ──────────────────────────────
                if mode == "cal" and eps_grid is not None:
                    for ei, eps in enumerate(eps_grid):
                        eps = float(eps)

                        # E_A blend
                        z_ea = ((1.0 - eps) * z_A + eps * z_DA)
                        p_ea = _softmax_numpy(z_ea)
                        cp, srank, ccov = _precompute_sorted(p_ea, ids, scrs)
                        eps_stats_EA[ei].update_cal(l, cp, srank, ccov, gamma_grid)
                        del cp, srank, ccov, p_ea, z_ea

                        # E_B blend
                        z_eb = ((1.0 - eps) * z_B + eps * z_DB)
                        p_eb = _softmax_numpy(z_eb)
                        cp, srank, ccov = _precompute_sorted(p_eb, ids, scrs)
                        eps_stats_EB[ei].update_cal(l, cp, srank, ccov, gamma_grid)
                        del cp, srank, ccov, p_eb, z_eb

                # ── Test: apply best ε, accumulate stats ──────────────────
                elif mode == "tst":
                    eps_a = float(best_eps_EA[l])
                    z_ea  = (1.0 - eps_a) * z_A + eps_a * z_DA
                    p_ea  = _softmax_numpy(z_ea)

                    eps_b = float(best_eps_EB[l])
                    z_eb  = (1.0 - eps_b) * z_B + eps_b * z_DB
                    p_eb  = _softmax_numpy(z_eb)

                    cp, srank, ccov = _precompute_sorted(p_ea, ids, scrs)
                    sig_EA_stats.update_tst(l, cp, srank, ccov)
                    del cp, srank, ccov

                    cp, srank, ccov = _precompute_sorted(p_eb, ids, scrs)
                    sig_EB_stats.update_tst(l, cp, srank, ccov)
                    del cp, srank, ccov

                    dual_stats.update(
                        l, p_ea, p_eb,
                        sig_EB_stats.gamma_stars[l],
                        ids, scrs,
                    )
                    del p_ea, z_ea, p_eb, z_eb

                del z_A, z_DA, z_B, z_DB

            # Save carry for next sequence's layer-0 Signal A
            last_h = tracer.h_in_list[n_layers - 1]
            if last_h is not None:
                carry_h = last_h[-1:, :].astype(np.float32).copy()  # (1, H)

            tracer.reset()

            if n_batches % 50 == 0:
                total = n_batches * seq_len * n_layers
                print(f"  [{desc}] batch={n_batches}  records={total}", flush=True)


# ════════════════════════════════════════════════════ main entry point

def run_gemma4_dual(
    alpha:       float          = 0.10,
    dataset_name: str           = "allenai/c4",
    dataset_config: str         = "en",
    split_cal:   str            = "train",
    split_test:  str            = "validation",
    cache_dir:   str            = "/datasets",
    max_samples_cal:  int       = 2000,
    max_samples_test: int       = 200,
    max_seq_len: int            = 512,
    text_field:  str            = "text",
    seed:        int            = 0,
    n_gamma:     int            = 400,
    eps_grid:    Optional[np.ndarray] = None,
) -> dict:
    """
    Two-phase dual-signal CRC prefetch pipeline for Gemma4-26B-A4B.

    Returns a dict with keys:
        best_eps_EA   : list[float], per-layer ε for E_A (rank-1 calibrated)
        best_eps_EB   : list[float], per-layer ε for E_B (rank-8 calibrated)
        gamma_stars   : list[float], per-layer γ* for E_B CRC
        metrics_EA    : rank-hit metrics for E_A (best-effort early prefetch)
        metrics_EB    : CRC set metrics for E_B (formal guarantee E[L]≤α)
        metrics_dual  : hybrid union metrics (overlap, avg_union, ...)
    """
    if eps_grid is None:
        eps_grid = np.array(
            [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            dtype=np.float32,
        )

    n_eps = len(eps_grid)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # ── 1. Load model (2 GPUs via device_map="auto") ──────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype = torch.bfloat16

    print(f"[g4d] Loading tokenizer from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)

    print(f"[g4d] Loading model → dtype=bfloat16, device_map=auto (2 GPUs)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[g4d] Model loaded.  Params: {total_params:.2f}B", flush=True)

    # ── 2. Locate text decoder layers ─────────────────────────────────────
    # Gemma4ForConditionalGeneration → .model (Gemma4Model)
    #   → .language_model (Gemma4TextModel) → .layers
    text_model = model.model.language_model
    n_layers   = len(text_model.layers)
    K          = 8          # top_k_experts (Gemma4-26B-A4B config)
    E          = 128        # num_experts
    print(f"[g4d] Text decoder: {n_layers} layers, K={K}, E={E}", flush=True)

    # ── 3. Extract router params as numpy (frees no GPU memory, just copies) ─
    print("[g4d] Extracting router parameters...", flush=True)
    router_params = extract_router_params(text_model)
    n_moe = sum(1 for rp in router_params if rp is not None)
    print(f"[g4d] {n_moe} / {n_layers} layers have MoE router.", flush=True)

    # ── 4. Dataset ─────────────────────────────────────────────────────────
    cal_data, test_data = load_splits(
        dataset_name, dataset_config, split_cal, split_test,
        cache_dir=cache_dir,
    )

    # ── 5. Attach tracer ───────────────────────────────────────────────────
    tracer = Gemma4DualTracer(text_model)
    print(f"[g4d] Hooked {n_layers} decoder layers (layer + router hooks).",
          flush=True)
    print(f"[g4d] eps_grid={list(eps_grid)}", flush=True)

    # ── 6. Gamma grid ──────────────────────────────────────────────────────
    gamma_grid = np.linspace(0.01, 0.9999, n_gamma)

    # ── 7. Calibration accumulators (one per ε, per signal path) ─────────
    # _StreamStats(n_pairs, n_gamma, K):  n_pairs = all n_layers (30)
    eps_stats_EA = [_StreamStats(n_layers, n_gamma, K) for _ in range(n_eps)]
    eps_stats_EB = [_StreamStats(n_layers, n_gamma, K) for _ in range(n_eps)]

    # ── 8. Calibration pass ────────────────────────────────────────────────
    print("\n[g4d] === Calibration pass (streaming — O(1) RAM) ===", flush=True)
    t0 = time.time()
    _gemma4_stream_pass(
        tracer, router_params, tokenizer, model,
        cal_data, max_samples_cal, max_seq_len, text_field, seed,
        desc="cal",
        eps_stats_EA=eps_stats_EA,
        eps_stats_EB=eps_stats_EB,
        gamma_grid=gamma_grid,
        eps_grid=eps_grid,
        mode="cal",
    )
    print(f"[g4d] Calibration done in {time.time() - t0:.0f}s", flush=True)
    n_cal = eps_stats_EA[0].cal_n_tok[0] if eps_stats_EA[0].cal_n_tok else 0
    print(f"[g4d] n_cal tokens per layer ≈ {n_cal}", flush=True)

    # ── 9. Find best ε per layer ───────────────────────────────────────────
    # E_A: maximise rank-1 hit rate
    best_eps_EA = []
    print("\n[g4d] Signal E_A — per-layer optimal ε (rank-1 hit rate):")
    print("[g4d]   (* row 29→00: cross-sequence carry — layer-29 output of prev seq)")
    hdr = "  Layer  " + "".join(f"  {e:.2f}" for e in eps_grid) + "  best_ε"
    print(hdr)
    for l in range(n_layers):
        r1_by_eps = []
        for ei in range(n_eps):
            n_l = eps_stats_EA[ei].cal_n_tok[l]
            r1  = float(eps_stats_EA[ei].cal_rh_sum[l][1]) / n_l if n_l > 0 else 0.0
            r1_by_eps.append(r1)
        best_ei = int(np.argmax(r1_by_eps))
        best_eps_EA.append(float(eps_grid[best_ei]))
        vals_str = "".join(f"  {v:.4f}" for v in r1_by_eps)
        # l=0: cross-sequence carry from layer (n_layers-1); l>0: previous layer
        src   = (l - 1) if l > 0 else (n_layers - 1)
        flag  = "*" if l == 0 else " "   # * = cross-sequence carry
        print(f"  {src:02d}→{l:02d}{flag}{vals_str}  ε={eps_grid[best_ei]:.2f}")

    # E_B: maximise rank-K hit rate
    best_eps_EB = []
    print(f"\n[g4d] Signal E_B — per-layer optimal ε (rank-{K} hit rate):")
    print(hdr)
    for l in range(n_layers):
        rK_by_eps = []
        for ei in range(n_eps):
            n_l  = eps_stats_EB[ei].cal_n_tok[l]
            rK   = float(eps_stats_EB[ei].cal_rh_sum[l][K]) / n_l if n_l > 0 else 0.0
            rK_by_eps.append(rK)
        best_ei = int(np.argmax(rK_by_eps))
        best_eps_EB.append(float(eps_grid[best_ei]))
        vals_str = "".join(f"  {v:.4f}" for v in rK_by_eps)
        print(f"  {l:02d}→{l:02d}{vals_str}  ε={eps_grid[best_ei]:.2f}")

    print(f"\n[g4d] Best ε per layer (E_A): {best_eps_EA}")
    print(f"[g4d] Best ε per layer (E_B): {best_eps_EB}")

    # ── 10. CRC calibration γ* for E_B (using best ε per layer) ──────────
    # Use the eps_stats_EB entry whose ε matches best_eps_EB[l] — the one that
    # was updated with the best ε's data during the cal pass.
    print("\n[g4d] Calibrating γ* for E_B CRC...")
    # For each layer, calibrate the _StreamStats corresponding to best_eps_EB[l]
    # We build a combined _StreamStats by selecting per layer from the right eps bin
    crc_stats = _StreamStats(n_layers, n_gamma, K)
    for l in range(n_layers):
        best_ei = int(np.argmin(np.abs(eps_grid - best_eps_EB[l])))
        crc_stats.cal_loss_sum[l] = eps_stats_EB[best_ei].cal_loss_sum[l].copy()
        crc_stats.cal_n_tok[l]    = eps_stats_EB[best_ei].cal_n_tok[l]
        crc_stats.cal_rh_sum[l]   = eps_stats_EB[best_ei].cal_rh_sum[l].copy()
    crc_stats.calibrate(alpha, n_cal, gamma_grid)
    gamma_stars = list(crc_stats.gamma_stars)

    print("[g4d] Per-layer γ* (E_B CRC):")
    for l in range(n_layers):
        print(f"  {l:02d}  γ*={gamma_stars[l]:.4f}")

    # ── 11. Test accumulators ──────────────────────────────────────────────
    sig_EA_stats = _StreamStats(n_layers, n_gamma, K)
    sig_EA_stats.gamma_stars = [1.0] * n_layers   # E_A needs no γ*

    sig_EB_stats = _StreamStats(n_layers, n_gamma, K)
    sig_EB_stats.gamma_stars = gamma_stars         # inject calibrated γ*

    dual_stats = _DualStats(n_layers, K)

    # ── 12. Test pass ──────────────────────────────────────────────────────
    print("\n[g4d] === Test pass (streaming) ===", flush=True)
    t0 = time.time()
    _gemma4_stream_pass(
        tracer, router_params, tokenizer, model,
        test_data, max_samples_test, max_seq_len, text_field, seed,
        desc="tst",
        eps_stats_EA=None,
        eps_stats_EB=None,
        gamma_grid=None,
        eps_grid=None,
        best_eps_EA=best_eps_EA,
        best_eps_EB=best_eps_EB,
        sig_EA_stats=sig_EA_stats,
        sig_EB_stats=sig_EB_stats,
        dual_stats=dual_stats,
        mode="tst",
    )
    print(f"[g4d] Test pass done in {time.time() - t0:.0f}s", flush=True)

    # ── 13. Collect results ────────────────────────────────────────────────
    tracer.remove()

    metrics_EA   = sig_EA_stats.results(K)
    metrics_EB   = sig_EB_stats.results(K)
    metrics_dual = dual_stats.results()

    # ── 14. Print summary ─────────────────────────────────────────────────
    print(f"\n{SEP}", flush=True)
    print(f"[g4d] GEMMA4 DUAL-SIGNAL RESULTS  (α={alpha}  K={K}  E={E}  "
          f"n_cal≈{n_cal})", flush=True)
    print(SEP, flush=True)

    if metrics_EA:
        rh = metrics_EA.get("rank_hit", {})
        print(
            f"  E_A (early, top-1):  "
            f"r1={rh.get(1, 0):.4f}  r2={rh.get(2, 0):.4f}  "
            f"r4={rh.get(4, 0):.4f}  r8={rh.get(8, 0):.4f}"
        )
    if metrics_EB:
        rh = metrics_EB.get("rank_hit", {})
        print(
            f"  E_B (CRC set):       "
            f"avg|S|={metrics_EB.get('avg_set', 0):.2f}  "
            f"E[L]={metrics_EB.get('mean_L', 0):.4f}  "
            f"r1={rh.get(1, 0):.4f}  r{K}={rh.get(K, 0):.4f}"
        )
    if metrics_dual:
        print(
            f"  Hybrid union:        "
            f"overlap={metrics_dual.get('overlap_rate', 0):.4f}  "
            f"avg|union|={metrics_dual.get('avg_union', 0):.2f}  "
            f"eb_coverage={metrics_dual.get('eb_coverage', 0):.4f}"
        )
    print(SEP, flush=True)

    return {
        "model":          "Gemma4-26B-A4B",
        "alpha":          alpha,
        "K":              K,
        "E":              E,
        "n_layers":       n_layers,
        "n_cal_tokens":   n_cal,
        "eps_grid":       list(eps_grid),
        "best_eps_EA":    best_eps_EA,
        "best_eps_EB":    best_eps_EB,
        "gamma_stars":    gamma_stars,
        "metrics_EA":     metrics_EA,
        "metrics_EB":     metrics_EB,
        "metrics_dual":   metrics_dual,
    }
