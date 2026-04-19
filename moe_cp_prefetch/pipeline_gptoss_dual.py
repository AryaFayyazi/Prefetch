"""
pipeline_gptoss_dual.py — Two-Phase Expert Prefetch Pipeline for openai/gpt-oss-20b
=====================================================================================

Design
------
All 24 GptOssDecoderLayer blocks contain a GptOssMLP with a GptOssTopKRouter.
The router receives post_attention_layernorm(h_post_attn_l), already reshaped
to (T, H) by GptOssMLP before calling the router.  So:

  Phase 1  (EARLY — fires after layer l-1 completes, before attention at layer l):
    Signal E_A — velocity-corrected router-logit blend using the previous layer's
                 full residual output as a proxy for the router input at layer l.

      h_in_l        = output of decoder layer l-1         (B,T,H) → squeezed (T,H)
      h_extrap_A    = 2·h_in_l − h_in_{l−1}              (cross-layer velocity)

      z_A    = router_logits_np(h_in_l,     rp_l)         (Signal A raw logits)
      z_DA   = router_logits_np(h_extrap_A, rp_l)
      z_EA   = (1−ε)·z_A + ε·z_DA  →  softmax(z_EA)

      ε chosen per-layer on calibration to maximise rank-1 hit rate.
      Prefetch: top-1(p_EA) — 1 expert, long load window.

    Layer 0 special case: no l-1 exists for the first sequence; instead we
    broadcast the last token of the previous sequence's layer-23 output as the
    carry (cross-sequence carry).  First sequence falls back to Signal B.

  Phase 2  (EXACT — fires when h_post_attn_l is computed, immediately before
            the MoE experts run):
    Signal E_B — velocity-corrected blend using the exact router input.

      h_pre_router_l  = post_attention_layernorm(h_post_attn_l), shape (T, H)
                        (captured via pre-hook on layer.mlp.router)
      h_extrap_B      = 2·h_pre_router_l − h_pre_router_{l−1}

      z_B    = router_logits_np(h_pre_router_l,  rp_l)
      z_DB   = router_logits_np(h_extrap_B,      rp_l)
      z_EB   = (1−ε)·z_B + ε·z_DB  →  softmax(z_EB)

      ε chosen per-layer on calibration to maximise rank-K hit rate (K=4).
      CRC conformal set: covers all K=4 true experts with E[L] ≤ α.

  Hybrid prefetch set = { top-1(E_A) } ∪ CRC-set(E_B)

Model constants — gpt-oss-20b:
  hidden_size  H = 2880   num_local_experts E = 32   num_experts_per_tok K = 4
  num_hidden_layers L = 24  (ALL 24 layers have MoE)
  Router: F.linear(h, weight, bias)  — no normalisation step
  Weights: mxfp4 (auto-dequantised to bf16 when kernels package not installed)
  Device:  device_map='auto' across 2 A6000 GPUs (~40 GB in bf16)
"""

from __future__ import annotations

import gc
import os
import time
from typing import List, Optional

import numpy as np
import torch

from moe_cp_prefetch.data import load_splits, iter_token_blocks
from moe_cp_prefetch.gptoss_adapter import (
    MODEL_PATH,
    GptOssDualTracer,
    GptOssRouterParams,
    extract_router_params,
    router_logits_np,
)

# Generic model-agnostic numpy helpers — reused across all model pipelines
from moe_cp_prefetch.pipeline_qwen_signals import (
    _StreamStats,
    _softmax_numpy,
    _precompute_sorted,
    _loss_at_gamma_single,
)

SEP = "═" * 110


# ════════════════════════════════════════════════════ _DualStats accumulator ══

class _DualStats:
    """Test-time accumulator for dual-signal (E_A + E_B) hybrid prefetch."""

    def __init__(self, n_layers: int, K: int):
        self.n_layers    = n_layers
        self.K           = K
        self.ea_r1_sum   = [0.0] * n_layers
        self.ea_r2_sum   = [0.0] * n_layers
        self.ea_r4_sum   = [0.0] * n_layers
        self.ea_r8_sum   = [0.0] * n_layers
        self.eb_set_sum  = [0.0] * n_layers
        self.eb_loss_sum = [0.0] * n_layers
        self.eb_cov_sum  = [0.0] * n_layers
        self.overlap_sum = [0.0] * n_layers
        self.union_sum   = [0.0] * n_layers
        self.n_tok       = [0]   * n_layers

    def update(self, l, p_EA, p_EB, gamma_EB, true_ids, true_scrs):
        T, E = p_EA.shape
        K    = true_ids.shape[1]

        ea_order = np.argsort(-p_EA, axis=1)
        ea_ranks = np.empty_like(ea_order)
        ea_ranks[np.arange(T)[:, None], ea_order] = np.arange(1, E + 1)[None, :]
        ea_true_ranks = ea_ranks[np.arange(T)[:, None], true_ids]

        for k_hit, store in [
            (1, self.ea_r1_sum), (2, self.ea_r2_sum),
            (4, self.ea_r4_sum), (8, self.ea_r8_sum),
        ]:
            store[l] += float((ea_true_ranks == k_hit).any(axis=1).sum())
        top1_EA = ea_order[:, 0]

        eb_order    = np.argsort(-p_EB, axis=1)
        eb_p_sorted = np.take_along_axis(p_EB, eb_order, axis=1)
        cum_eb      = np.cumsum(eb_p_sorted, axis=1)
        k_stars     = np.minimum((cum_eb[:, :-1] < gamma_EB).sum(axis=1) + 1, E)

        eb_ranks = np.empty_like(eb_order)
        eb_ranks[np.arange(T)[:, None], eb_order] = np.arange(1, E + 1)[None, :]
        set_mask_EB = eb_ranks <= k_stars[:, np.newaxis]
        eb_set_size = set_mask_EB.sum(axis=1)
        self.eb_set_sum[l] += float(eb_set_size.sum())

        eb_true_ranks = eb_ranks[np.arange(T)[:, None], true_ids]
        sort_k = np.argsort(eb_true_ranks, axis=1)
        sorted_true_ranks = np.take_along_axis(eb_true_ranks, sort_k, axis=1)
        sorted_true_scrs  = np.take_along_axis(true_scrs,    sort_k, axis=1)
        cum_covered = np.cumsum(sorted_true_scrs, axis=1)
        _, losses = _loss_at_gamma_single(
            eb_p_sorted, sorted_true_ranks, cum_covered, gamma_EB,
        )
        self.eb_loss_sum[l] += float(losses.sum())

        all_covered = np.ones(T, dtype=bool)
        for k_idx in range(K):
            all_covered &= set_mask_EB[np.arange(T), true_ids[:, k_idx]]
        self.eb_cov_sum[l] += float(all_covered.sum())

        overlap  = set_mask_EB[np.arange(T), top1_EA]
        union_sz = np.where(overlap, eb_set_size, eb_set_size + 1)
        self.overlap_sum[l] += float(overlap.sum())
        self.union_sum[l]   += float(union_sz.sum())
        self.n_tok[l] += T

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


# ════════════════════════════════════════════════════════ streaming pass ══════

def _gptoss_stream_pass(
    tracer:         GptOssDualTracer,
    router_params:  List[Optional[GptOssRouterParams]],
    tokenizer,
    model:          torch.nn.Module,
    split_data,
    max_samples:    int,
    max_seq_len:    int,
    text_field:     str,
    seed:           int,
    desc:           str,
    # ── calibration accumulators (mode="cal") ──────────────────────────────
    eps_stats_EA:   Optional[List[_StreamStats]],
    eps_stats_EB:   Optional[List[_StreamStats]],
    gamma_grid:     Optional[np.ndarray],
    eps_grid:       Optional[np.ndarray],
    # ── test accumulators (mode="tst") ─────────────────────────────────────
    best_eps_EA:    Optional[List[float]] = None,
    best_eps_EB:    Optional[List[float]] = None,
    sig_EA_stats:   Optional[_StreamStats] = None,
    sig_EB_stats:   Optional[_StreamStats] = None,
    dual_stats:     Optional[_DualStats]   = None,
    mode:           str = "cal",
) -> None:
    """
    Single streaming pass (calibration OR test) over the dataset.

    For each sequence:
      1. Run model forward (hooks populate tracer in-place).
      2. For every MoE layer l:
           - Compute z_B, z_DB  from h_pre_router[l]  (Signal B exact).
           - Compute z_A, z_DA  from h_in_list[l-1]   (Signal A early, l≥1)
             or from cross-sequence carry              (l=0, seqs after the first).
           - In cal mode:  sweep all eps, update eps_stats_EA/EB.
           - In tst mode:  apply best eps, update sig_EA_stats / sig_EB_stats /
                           dual_stats.
      3. Save cross-sequence carry (layer-23 output last token).
      4. Reset tracer.
    """
    n_layers = tracer.n_layers   # 24 for gpt-oss-20b
    n_eps    = len(eps_grid) if eps_grid is not None else 0
    n_batches = 0

    # Cross-sequence carry for layer 0 Signal A:
    # After each sequence we store h_in_list[n_layers-1][-1:] (last token,
    # shape (1, H)).  At l=0 of the *next* sequence we broadcast it to (T, H).
    carry_h: Optional[np.ndarray] = None   # (1, H) float32, updated each seq

    with torch.no_grad():
        for batch in iter_token_blocks(
            split_data, tokenizer, text_field,
            max_samples, max_seq_len, seed,
            batch_size=1,
        ):
            ids_t  = torch.tensor(batch.input_ids,      dtype=torch.long)
            mask_t = torch.tensor(batch.attention_mask, dtype=torch.long)

            # device_map='auto' places the embed layer on the first GPU;
            # move input ids there
            first_device = next(model.parameters()).device
            _ = model(
                input_ids=ids_t.to(first_device),
                attention_mask=mask_t.to(first_device),
                use_cache=False,
            )

            n_batches += 1
            seq_len = ids_t.shape[1]

            # ── per-layer signal computation ──────────────────────────────
            for l in range(n_layers):
                rp = router_params[l]
                if rp is None:
                    continue

                ids  = tracer.true_topk_ids[l]   # (T, K) or None
                scrs = tracer.true_topk_scrs[l]  # (T, K) or None
                if ids is None or scrs is None:
                    continue

                h_post = tracer.h_pre_router[l]  # (T, H) — exact router input
                if h_post is None:
                    continue

                T = h_post.shape[0]

                # ── Signal B logits (exact pre-router hidden state) ───────
                z_B = router_logits_np(h_post, rp)   # (T, E)

                # ── Signal D_B logits (cross-layer velocity on pre-router) ─
                h_post_prev = tracer.h_pre_router[l - 1] if l > 0 else None
                if h_post_prev is not None and h_post_prev.shape[0] == T:
                    z_DB = router_logits_np(
                        (2.0 * h_post - h_post_prev).astype(np.float32), rp
                    )
                else:
                    z_DB = z_B                       # no velocity; fallback

                # ── Signal A logits (previous layer output as proxy) ──────
                #    l > 0: use output of layer l-1 (shape already (T, H))
                #    l = 0: use cross-sequence carry (last seq's layer-23 last tok)
                h_in_l   = tracer.h_in_list[l - 1] if l > 0 else None
                h_in_lm1 = tracer.h_in_list[l - 2] if l > 1 else None

                # Layer-0 carry: broadcast previous sequence's final residual
                if h_in_l is None and carry_h is not None:
                    h_in_l = np.ascontiguousarray(
                        np.broadcast_to(carry_h, (T, carry_h.shape[1]))
                    ).astype(np.float32)
                    # h_in_lm1 stays None → z_DA = z_A (no velocity at l=0)

                has_ea = h_in_l is not None

                if has_ea:
                    h_in_l = h_in_l.astype(np.float32)
                    z_A = router_logits_np(h_in_l, rp)
                    if h_in_lm1 is not None and h_in_lm1.shape[0] == T:
                        z_DA = router_logits_np(
                            (2.0 * h_in_l - h_in_lm1.astype(np.float32)), rp
                        )
                    else:
                        z_DA = z_A                   # no velocity; fallback
                else:
                    z_A = z_B; z_DA = z_DB           # first seq l=0: use B

                # ── Calibration: sweep all ε ──────────────────────────────
                if mode == "cal" and eps_grid is not None:
                    for ei, eps in enumerate(eps_grid):
                        eps = float(eps)

                        z_ea = (1.0 - eps) * z_A + eps * z_DA
                        p_ea = _softmax_numpy(z_ea)
                        cp, srank, ccov = _precompute_sorted(p_ea, ids, scrs)
                        eps_stats_EA[ei].update_cal(l, cp, srank, ccov, gamma_grid)
                        del cp, srank, ccov, p_ea, z_ea

                        z_eb = (1.0 - eps) * z_B + eps * z_DB
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


# ════════════════════════════════════════════════════ main entry point ════════

def run_gptoss_dual(
    alpha:            float          = 0.10,
    dataset_name:     str            = "allenai/c4",
    dataset_config:   str            = "en",
    split_cal:        str            = "train",
    split_test:       str            = "validation",
    cache_dir:        str            = "/datasets",
    max_samples_cal:  int            = 2000,
    max_samples_test: int            = 200,
    max_seq_len:      int            = 512,
    text_field:       str            = "text",
    seed:             int            = 0,
    n_gamma:          int            = 400,
    eps_grid:         Optional[np.ndarray] = None,
) -> dict:
    """
    Two-phase dual-signal CRC prefetch pipeline for openai/gpt-oss-20b.

    Returns a dict with keys:
        model         : "gpt-oss-20b"
        alpha         : float
        K, E, n_layers: int
        n_cal_tokens  : int
        eps_grid      : list[float]
        best_eps_EA   : list[float], per-layer ε for E_A (rank-1 calibrated)
        best_eps_EB   : list[float], per-layer ε for E_B (rank-K calibrated)
        gamma_stars   : list[float], per-layer γ* for E_B CRC
        metrics_EA    : rank-hit metrics for E_A
        metrics_EB    : CRC set metrics for E_B
        metrics_dual  : hybrid union metrics
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

    print(f"[gos] Loading tokenizer from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"[gos] Loading model → dtype=bfloat16, device_map=auto (2 GPUs)", flush=True)
    print(f"[gos] Note: experts are mxfp4-quantised; dequantised to bf16 if "
          f"'kernels' package is not installed.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[gos] Model loaded.  Params: {total_params:.2f}B", flush=True)

    # ── 2. Locate decoder layers ───────────────────────────────────────────
    # GptOssForCausalLM → .model (GptOssModel) → .layers
    n_layers = len(model.model.layers)
    K        = 4    # num_experts_per_tok
    E        = 32   # num_local_experts
    H        = 2880 # hidden_size
    print(f"[gos] Decoder: {n_layers} layers, K={K}, E={E}, H={H}", flush=True)

    # ── 3. Extract router params as float32 numpy ──────────────────────────
    print("[gos] Extracting router parameters...", flush=True)
    router_params = extract_router_params(model)
    n_moe = sum(1 for rp in router_params if rp is not None)
    print(f"[gos] {n_moe} / {n_layers} layers have router params.", flush=True)

    # ── 4. Dataset ─────────────────────────────────────────────────────────
    cal_data, test_data = load_splits(
        dataset_name, dataset_config, split_cal, split_test,
        cache_dir=cache_dir,
    )

    # ── 5. Attach tracer ───────────────────────────────────────────────────
    tracer = GptOssDualTracer(model)
    print(f"[gos] Hooked {n_layers} decoder layers (layer + router hooks).",
          flush=True)
    print(f"[gos] eps_grid={list(eps_grid)}", flush=True)

    # ── 6. Gamma grid ──────────────────────────────────────────────────────
    gamma_grid = np.linspace(0.01, 0.9999, n_gamma)

    # ── 7. Calibration accumulators ────────────────────────────────────────
    eps_stats_EA = [_StreamStats(n_layers, n_gamma, K) for _ in range(n_eps)]
    eps_stats_EB = [_StreamStats(n_layers, n_gamma, K) for _ in range(n_eps)]

    # ── 8. Calibration pass ────────────────────────────────────────────────
    print(f"\n[gos] === Calibration pass (streaming — O(1) RAM) ===", flush=True)
    t0 = time.time()
    _gptoss_stream_pass(
        tracer, router_params, tokenizer, model,
        cal_data, max_samples_cal, max_seq_len, text_field, seed,
        desc="cal",
        eps_stats_EA=eps_stats_EA,
        eps_stats_EB=eps_stats_EB,
        gamma_grid=gamma_grid,
        eps_grid=eps_grid,
        mode="cal",
    )
    print(f"[gos] Calibration done in {time.time() - t0:.0f}s", flush=True)
    n_cal = eps_stats_EA[0].cal_n_tok[0] if eps_stats_EA[0].cal_n_tok else 0
    print(f"[gos] n_cal tokens per layer ≈ {n_cal}", flush=True)

    # ── 9. Find best ε per layer ───────────────────────────────────────────
    best_eps_EA = []
    print(f"\n[gos] Signal E_A — per-layer optimal ε (rank-1 hit rate):")
    print(f"[gos]   (* row {n_layers-1:02d}→00: cross-sequence carry)")
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
        src  = (l - 1) if l > 0 else (n_layers - 1)
        flag = "*" if l == 0 else " "
        print(f"  {src:02d}→{l:02d}{flag}{vals_str}  ε={eps_grid[best_ei]:.2f}")

    best_eps_EB = []
    print(f"\n[gos] Signal E_B — per-layer optimal ε (rank-{K} hit rate):")
    print(hdr)
    for l in range(n_layers):
        rK_by_eps = []
        for ei in range(n_eps):
            n_l = eps_stats_EB[ei].cal_n_tok[l]
            rK  = float(eps_stats_EB[ei].cal_rh_sum[l][K]) / n_l if n_l > 0 else 0.0
            rK_by_eps.append(rK)
        best_ei = int(np.argmax(rK_by_eps))
        best_eps_EB.append(float(eps_grid[best_ei]))
        vals_str = "".join(f"  {v:.4f}" for v in rK_by_eps)
        print(f"  {l:02d}→{l:02d}{vals_str}  ε={eps_grid[best_ei]:.2f}")

    print(f"\n[gos] Best ε per layer (E_A): {best_eps_EA}")
    print(f"[gos] Best ε per layer (E_B): {best_eps_EB}")

    # ── 10. CRC calibration γ* for E_B ────────────────────────────────────
    print(f"\n[gos] Calibrating γ* for E_B CRC (α={alpha})...", flush=True)
    crc_stats = _StreamStats(n_layers, n_gamma, K)
    for l in range(n_layers):
        best_ei = int(np.argmin(np.abs(eps_grid - best_eps_EB[l])))
        crc_stats.cal_loss_sum[l] = eps_stats_EB[best_ei].cal_loss_sum[l].copy()
        crc_stats.cal_n_tok[l]    = eps_stats_EB[best_ei].cal_n_tok[l]
        crc_stats.cal_rh_sum[l]   = eps_stats_EB[best_ei].cal_rh_sum[l].copy()
    crc_stats.calibrate(alpha, n_cal, gamma_grid)
    gamma_stars = list(crc_stats.gamma_stars)

    print(f"[gos] E_B γ* per layer:")
    for l in range(n_layers):
        print(f"  {l:02d}  γ*={gamma_stars[l]:.4f}")

    # ── 11. Test accumulators ──────────────────────────────────────────────
    sig_EA_stats = _StreamStats(n_layers, n_gamma, K)
    sig_EA_stats.gamma_stars = [1.0] * n_layers

    sig_EB_stats = _StreamStats(n_layers, n_gamma, K)
    sig_EB_stats.gamma_stars = gamma_stars

    dual_stats = _DualStats(n_layers, K)

    # ── 12. Test pass ──────────────────────────────────────────────────────
    print(f"\n[gos] === Test pass (streaming) ===", flush=True)
    t0 = time.time()
    _gptoss_stream_pass(
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
    print(f"[gos] Test pass done in {time.time() - t0:.0f}s", flush=True)

    # ── 13. Collect results ────────────────────────────────────────────────
    tracer.remove()

    metrics_EA   = sig_EA_stats.results(K)
    metrics_EB   = sig_EB_stats.results(K)
    metrics_dual = dual_stats.results()

    # ── 14. Print summary ─────────────────────────────────────────────────
    print(f"\n{SEP}", flush=True)
    print(f"[gos] GPT-OSS-20B DUAL-SIGNAL RESULTS  (α={alpha}  K={K}  E={E}  "
          f"n_cal≈{n_cal})", flush=True)
    print(SEP, flush=True)

    if metrics_EA:
        rh = metrics_EA.get("rank_hit", {})
        print(
            f"  E_A (early, top-1):  "
            f"r1={rh.get(1, 0):.4f}  r2={rh.get(2, 0):.4f}  "
            f"r4={rh.get(4, 0):.4f}"
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

    print(f"\n[gos] Per-layer results:")
    print(f"  {'Layer':>5}   {'E_A r1':>8}   {'E_B |S|':>8}   "
          f"{'E_B cov':>8}   {'overlap':>8}   {'union':>6}   "
          f"{'bestε_A':>7}   {'bestε_B':>7}")
    dual_pl = metrics_dual.get("per_layer", {}) if metrics_dual else {}
    for l in range(n_layers):
        du  = dual_pl.get(l, {})
        src  = (l - 1) if l > 0 else (n_layers - 1)
        flag = "*" if l == 0 else " "
        print(
            f"  {src:02d}→{l:02d}{flag}   "
            f"{du.get('ea_r1',      0):>8.4f}   "
            f"{du.get('avg_eb_set', 0):>8.2f}   "
            f"{du.get('eb_coverage',0):>8.4f}   "
            f"{du.get('overlap',    0):>8.4f}   "
            f"{du.get('avg_union',  0):>6.2f}   "
            f"{best_eps_EA[l]:>7.2f}   "
            f"{best_eps_EB[l]:>7.2f}"
        )
    print(SEP, flush=True)

    return {
        "model":         "gpt-oss-20b",
        "alpha":         alpha,
        "K":             K,
        "E":             E,
        "H":             H,
        "n_layers":      n_layers,
        "n_cal_tokens":  n_cal,
        "eps_grid":      list(eps_grid),
        "best_eps_EA":   best_eps_EA,
        "best_eps_EB":   best_eps_EB,
        "gamma_stars":   gamma_stars,
        "metrics_EA":    metrics_EA,
        "metrics_EB":    metrics_EB,
        "metrics_dual":  metrics_dual,
    }
