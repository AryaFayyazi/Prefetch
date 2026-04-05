"""
Qwen1.5-MoE-A2.7B  — Novel Signal Comparison Pipeline  (streaming, O(1) RAM)
═══════════════════════════════════════════════════════════════════════════════

Motivation
──────────
The baseline signal (h_post_attn_l → input_LN_{l+1} → W_gate_{l+1}) achieves
rank-4 = 0.41.  It operates on the wrong hidden state and the wrong LayerNorm:

  1. Wrong source h: h_post_attn_l is captured BEFORE MoE_l runs.
     The true input to layer l+1 is
       h_in_{l+1} = h_post_attn_l + MoE_l( post_attn_LN_l(h_post_attn_l) )
     The MoE output is a routing-dependent transformation of the hidden state.

  2. Wrong LayerNorm: The gate at layer l+1 applies post_attention_LN_{l+1}
     (the second LN of layer l+1, immediately before the MoE block).
     The baseline applies input_LN_{l+1} (the first LN, before self-attention)
     which has different trained scale/bias — an incorrect projection basis.

Three novel signals
────────────────────
  Signal B — Post-MoE + Correct LN
    Source : h_in_{l+1}  =  exact layer-boundary hidden state of layer l+1
             captured via pre-hook on input_layernorm of layer l+1.
    LN     : post_attention_LN_{l+1}  (the LN the gate actually uses).
    p_B    = softmax( W_gate_{l+1} @ post_attn_LN_{l+1}( h_in_{l+1} ) )
    Fixes both errors simultaneously.

  Signal C — Causal Routing Momentum (EMA prior)
    Consecutive tokens in a sequence activate similar experts. Build a causal
    EMA from the PREDICTED top-K of Signal B (no oracle, fully inference-time):
      ema[0]  = uniform(1/E)
      ema[t]  = (1−β)·ema[t−1] + β · one_hot_topK( rank-sorted p_B[t−1] )
    Blend:
      p_C[t] = normalise( (1−λ)·p_B[t]  +  λ·ema[t] )
    Reset at each new sequence.  Exploits intra-sequence routing coherence.

  Signal D — Residual velocity extrapolation
    The residual stream follows a trajectory across layers.  Linear momentum:
      h_extrap = 2·h_in_{l+1} − h_in_l
    p_D = softmax( W_gate_{l+1} @ post_attn_LN_{l+1}( h_extrap ) )
    Analogous to heavy-ball momentum in optimisation.

  Signal BD — Linear blend
    p_BD = normalise( bd_alpha·p_B  +  (1−bd_alpha)·p_D )

Memory design: STREAMING
────────────────────────
  The old implementation accumulated (T_total, E) prob arrays per signal per
  layer pair (5 signals × 23 layers × 1M tokens × 60 experts × 4 bytes ≈ 28 GB
  CPU RAM), causing OOM at 48 GB.

  This pipeline never stores large prob arrays.  Instead, for each mini-batch:
    1. Compute probs for all 5 signals  (T=512, E=60) — small, immediate.
    2. Compute weighted-compute losses at ALL n_gamma grid points in one
       vectorised call → (G, T);  immediately reduce to (G,) running sums.
    3. Compute rank-hit / coverage indicators → immediately reduce to scalars.
    4. Delete probs and intermediate tensors before the next layer pair.

  Peak extra RAM per batch per layer: ~15 MB (while evaluating one signal).

CRC guarantee (unchanged)
──────────────────────────
  E[ L(S_{γ*}, x_test) ] ≤ α + 1/(n+1)   where L = 1 − Σ_{e∈true_K∩S} g_e
"""
from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import ExperimentConfig
from .data import load_splits, iter_token_blocks
from .qwen_moe_postattn_adapter import QwenMoePostAttnTracer, PostAttnTrace


# ──────────────────────────────────────────── gate prediction helpers

def _gate_logits(
    h:        torch.Tensor,   # (T, H)  cpu  — any float
    W_gate:   torch.Tensor,   # (E, H)  model device
    ln_module,                # nn.Module (RMSNorm on model device)
) -> np.ndarray:              # (T, E)  numpy float32, cpu  — RAW logits (pre-softmax)
    """W_gate @ LN(h)  — returns raw gate logits, no softmax."""
    with torch.no_grad():
        hd       = h.to(W_gate.device, dtype=W_gate.dtype)
        h_normed = ln_module(hd)
        logits   = (h_normed @ W_gate.T).float()
    return logits.cpu().numpy()


def _gate_probs(
    h:        torch.Tensor,   # (T, H)  cpu  — any float
    W_gate:   torch.Tensor,   # (E, H)  model device
    ln_module,                # nn.Module (RMSNorm on model device)
) -> np.ndarray:              # (T, E)  numpy float32, cpu
    """softmax( W_gate @ LN(h) )"""
    z = _gate_logits(h, W_gate, ln_module)   # (T, E)
    # stable softmax
    z -= z.max(axis=1, keepdims=True)
    e  = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


# ──────────────────────────────────────────── causal EMA prior  (Signal C)

def _compute_ema_prior(
    p_B:    np.ndarray,  # (T, E)
    top_k:  int,
    beta:   float = 0.30,
    lam:    float = 0.30,
) -> np.ndarray:          # (T, E)  normalised
    """
    Causal routing-momentum prior built from Signal B's PREDICTED top-K.
    Fully causal — token t uses only predictions from tokens 0 … t-1.
    Reset at each new sequence (call once per sequence/batch).
    """
    T, E  = p_B.shape
    one_hots = np.zeros((T, E), dtype=np.float32)
    topk_idx = np.argpartition(-p_B, top_k, axis=1)[:, :top_k]  # (T, K) approx
    for t in range(T):
        one_hots[t, topk_idx[t]] = 1.0 / top_k

    ema  = np.full(E, 1.0 / E, dtype=np.float32)   # start uniform
    p_C  = np.empty((T, E), dtype=np.float32)

    for t in range(T):
        blended   = (1.0 - lam) * p_B[t] + lam * ema
        blended  /= blended.sum()
        p_C[t]    = blended
        ema       = (1.0 - beta) * ema + beta * one_hots[t]

    return p_C


# ─────────────────────────────── vectorised CRC core  (streaming version)

def _precompute_sorted(
    probs:       np.ndarray,   # (T, E)
    true_ids:    np.ndarray,   # (T, K)
    true_scores: np.ndarray,   # (T, K) renorm gate weights
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-time setup per (probs, true_ids, true_scores) triple.
    Returns:
      cum_pred          (T, E)  cumulative predicted mass (desc order)
      sorted_true_ranks (T, K)  predicted ranks of true experts (asc = best first)
      cum_covered       (T, K)  cumulative gate weight of true experts in rank order
    """
    T, E = probs.shape
    K    = true_ids.shape[1]

    order   = np.argsort(-probs, axis=1)           # (T, E)
    sp      = np.take_along_axis(probs, order, 1)  # (T, E) desc
    cum_pred = np.cumsum(sp, axis=1)               # (T, E)

    rank_all = np.empty((T, E), dtype=np.int32)
    rank_all[np.arange(T)[:, None], order] = np.arange(1, E + 1, dtype=np.int32)

    true_ranks = rank_all[np.arange(T)[:, None], true_ids]  # (T, K)
    rs         = np.argsort(true_ranks, axis=1)
    sorted_true_ranks = np.take_along_axis(true_ranks,   rs, 1)  # (T, K) asc
    sorted_scores     = np.take_along_axis(true_scores,  rs, 1)  # (T, K)
    cum_covered       = np.cumsum(sorted_scores, axis=1)          # (T, K)

    return cum_pred, sorted_true_ranks, cum_covered


def _loss_at_all_gammas(
    cum_pred:          np.ndarray,   # (T, E)
    sorted_true_ranks: np.ndarray,   # (T, K)
    cum_covered:       np.ndarray,   # (T, K)
    gamma_grid:        np.ndarray,   # (G,)
) -> np.ndarray:                     # (G, T) float32
    """
    Vectorised weighted-compute loss for every gamma in grid, every token.

    k*(t, γ) = min k s.t. cum_pred[t, k-1] >= γ  (clipped to E)
    L(t, γ)  = 1 − Σ_{e∈true_K, rank_e ≤ k*(t,γ)} g_e

    Memory: peak ~15 MB for typical G=400, T=512, E=60.
    """
    T, E = cum_pred.shape
    K    = sorted_true_ranks.shape[1]
    G    = len(gamma_grid)

    # k_stars (G, T)
    # cum_pred (T, E) → compare against each gamma
    # Use (G, 1, 1) broadcast against (1, T, E) → (G, T, E) bool
    k_stars = np.minimum(
        (cum_pred[np.newaxis, :, :] < gamma_grid[:, np.newaxis, np.newaxis]
         ).sum(axis=2, dtype=np.int32) + 1,
        E,
    )  # (G, T)

    # in_set_count (G, T): how many true experts fall within top-k*(t)
    # sorted_true_ranks (T, K) vs k_stars (G, T)
    in_set = (
        sorted_true_ranks[np.newaxis, :, :] <= k_stars[:, :, np.newaxis]
    ).sum(axis=2)  # (G, T) int

    safe_idx = np.maximum(in_set - 1, 0)  # (G, T)

    # covered = cum_covered[t, safe_idx[g, t]]
    T_idx   = np.broadcast_to(np.arange(T)[np.newaxis, :], (G, T))  # (G, T)
    covered = cum_covered[T_idx, safe_idx]                           # (G, T)
    covered = np.where(in_set > 0, covered, 0.0)

    return (1.0 - covered).astype(np.float32)  # (G, T)


def _loss_at_gamma_single(
    cum_pred:          np.ndarray,  # (T, E)
    sorted_true_ranks: np.ndarray,  # (T, K)
    cum_covered:       np.ndarray,  # (T, K)
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-token set sizes and losses at a single calibrated gamma.  Test time."""
    T, E = cum_pred.shape
    k_stars = np.minimum(
        (cum_pred < gamma).sum(axis=1, dtype=np.int32) + 1, E
    )  # (T,)
    in_set   = (sorted_true_ranks <= k_stars[:, np.newaxis]).sum(axis=1)  # (T,)
    safe_idx = np.maximum(in_set - 1, 0)
    covered  = np.where(in_set > 0, cum_covered[np.arange(T), safe_idx], 0.0)
    return k_stars.astype(np.int32), (1.0 - covered).astype(np.float32)


# ─────────────────────────────────── running-stat accumulators

class _StreamStats:
    """Per-signal per-layer streaming statistics.  Updated batch-by-batch."""

    def __init__(self, n_pairs: int, n_gamma: int, top_k: int):
        self.n_pairs = n_pairs
        self.K       = top_k

        # calibration: loss sums over gamma grid
        self.cal_loss_sum  = [np.zeros(n_gamma, dtype=np.float64) for _ in range(n_pairs)]
        self.cal_n_tok     = [0] * n_pairs

        # calibration: rank metrics
        self.cal_rh_sum    = [np.zeros(top_k + 1, dtype=np.float64) for _ in range(n_pairs)]
        self.cal_cov_sum   = [np.zeros(top_k + 1, dtype=np.float64) for _ in range(n_pairs)]

        # test: metrics accumulated after calibration γ* is set
        self.gamma_stars   = [1.0] * n_pairs
        self.tst_set_sum   = [0.0] * n_pairs
        self.tst_loss_sum  = [0.0] * n_pairs
        self.tst_full_cov  = [0.0] * n_pairs
        self.tst_rh_sum    = [np.zeros(top_k + 1, dtype=np.float64) for _ in range(n_pairs)]
        self.tst_cov_sum   = [np.zeros(top_k + 1, dtype=np.float64) for _ in range(n_pairs)]
        self.tst_n_tok     = [0] * n_pairs

    def update_cal(
        self,
        l:                 int,
        cum_pred:          np.ndarray,   # (T, E)
        sorted_true_ranks: np.ndarray,   # (T, K)
        cum_covered:       np.ndarray,   # (T, K)
        gamma_grid:        np.ndarray,   # (G,)
    ):
        T = cum_pred.shape[0]
        losses = _loss_at_all_gammas(
            cum_pred, sorted_true_ranks, cum_covered, gamma_grid
        )  # (G, T)
        self.cal_loss_sum[l] += losses.sum(axis=1)   # (G,)
        self.cal_n_tok[l]    += T

        K = self.K
        for k in range(1, K + 1):
            self.cal_rh_sum[l][k]  += float((sorted_true_ranks == k).any(axis=1).sum())
            self.cal_cov_sum[l][k] += float(((sorted_true_ranks <= k).sum(axis=1) == k).sum())

    def calibrate(self, alpha: float, n_cal: int, gamma_grid: np.ndarray):
        """Find γ* per layer (minimum γ s.t. mean_L + 1/(n+1) ≤ α)."""
        correction = 1.0 / (n_cal + 1)
        target     = alpha - correction
        for l in range(self.n_pairs):
            n = self.cal_n_tok[l]
            if n == 0:
                self.gamma_stars[l] = float(gamma_grid[-1])
                continue
            mean_losses = self.cal_loss_sum[l] / n   # (G,)
            # sweep small→large, first γ satisfying constraint
            idxs = np.where(mean_losses <= target)[0]
            if len(idxs) > 0:
                self.gamma_stars[l] = float(gamma_grid[idxs[0]])
            else:
                self.gamma_stars[l] = float(gamma_grid[-1])

    def update_tst(
        self,
        l:                 int,
        cum_pred:          np.ndarray,
        sorted_true_ranks: np.ndarray,
        cum_covered:       np.ndarray,
    ):
        g = self.gamma_stars[l]
        ss, ls = _loss_at_gamma_single(cum_pred, sorted_true_ranks, cum_covered, g)
        T = cum_pred.shape[0]
        K = self.K

        self.tst_set_sum[l]  += float(ss.sum())
        self.tst_loss_sum[l] += float(ls.sum())

        worst_rank = sorted_true_ranks[:, -1]
        self.tst_full_cov[l] += float((worst_rank <= ss).sum())

        for k in range(1, K + 1):
            self.tst_rh_sum[l][k]  += float((sorted_true_ranks == k).any(axis=1).sum())
            self.tst_cov_sum[l][k] += float(((sorted_true_ranks <= k).sum(axis=1) == k).sum())

        self.tst_n_tok[l] += T

    def results(self, top_k: int) -> dict:
        """Aggregate final metrics dict after the test pass."""
        valid = [l for l in range(self.n_pairs) if self.tst_n_tok[l] > 0]
        if not valid:
            return {}

        N_total = sum(self.tst_n_tok[l] for l in valid)

        avg_set  = sum(self.tst_set_sum[l]  for l in valid) / N_total
        mean_L   = sum(self.tst_loss_sum[l] for l in valid) / N_total
        p_full   = sum(self.tst_full_cov[l] for l in valid) / N_total
        p_any    = 1.0 - p_full

        rh   = {}
        cv   = {}
        cal_rh_agg = {}
        cal_cv_agg = {}

        for k in range(1, top_k + 1):
            rh[k] = sum(self.tst_rh_sum[l][k]  for l in valid) / N_total
            cv[k] = sum(self.tst_cov_sum[l][k] for l in valid) / N_total

        N_cal = sum(self.cal_n_tok[l] for l in valid)
        for k in range(1, top_k + 1):
            cal_rh_agg[k] = (
                sum(self.cal_rh_sum[l][k] for l in valid) / N_cal if N_cal else 0.0
            )
            cal_cv_agg[k] = (
                sum(self.cal_cov_sum[l][k] for l in valid) / N_cal if N_cal else 0.0
            )

        return {
            "avg_set":       avg_set,
            "mean_L":        mean_L,
            "p_full":        p_full,
            "p_any_miss":    p_any,
            "rank_hit":      rh,
            "cov_at":        cv,
            "cal_rank_hit":  cal_rh_agg,
            "cal_cov_at":    cal_cv_agg,
            "gamma_stars":   list(self.gamma_stars),
        }


# ─────────────────────── main forward pass  (one pass, 5+E signals, streaming)

_SIG_NAMES = ["A", "B", "C", "D", "BD"]


def _softmax_numpy(z: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax on a (T, E) array."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _stream_pass(
    tracer:              QwenMoePostAttnTracer,
    tokenizer,
    model,
    device,
    split_data,
    cfg:                 ExperimentConfig,
    max_samples:         int,
    desc:                str,
    stats:               List[_StreamStats],   # one per signal A/B/C/D/BD
    gamma_grid:          np.ndarray,
    mode:                str,                  # "cal" | "tst"
    ema_beta:            float,
    ema_lambda:          float,
    bd_alpha:            float,
    # ── Signal E (logit-blend sweep) ──────────────────────────────────────
    eps_grid:            Optional[np.ndarray]       = None,  # (Ne,) ε values to sweep (cal)
    eps_stats:           Optional[List[_StreamStats]] = None, # len=Ne, cal mode only
    best_eps_per_layer:  Optional[List[float]]      = None,  # per-layer ε (tst mode)
    sig_E_stats:         Optional[_StreamStats]     = None,  # tst mode accumulator
):
    """
    Single forward pass that streams statistics for all signals.

    Cal mode: updates stats[A/B/C/D/BD] AND eps_stats[i] for each ε in eps_grid.
    Tst mode: updates stats[A/B/C/D/BD] AND sig_E_stats using best_eps_per_layer.

    Signal E logit blend:
      z_ε(t) = (1−ε)·z_B(t) + ε·z_D(t)   (logit space)
      p_E(t) = softmax( z_ε(t) )
    where z_B = LN_pa(h_in_{l+1})·W^T  and  z_D = LN_pa(h_extrap)·W^T.
    Since the LN is identical, this is equivalent to:
      p_E(t) = softmax( W^T · LN_pa( (1−ε)·h_in_{l+1} + ε·h_extrap ) )   [approx, LN nonlinear]
    — a velocity-corrected gate prediction.
    """
    n_layers = tracer.num_layers()
    n_pairs  = n_layers - 1
    K        = cfg.router_top_k

    n_batches = 0
    with torch.no_grad():
        for batch in iter_token_blocks(
            split_data, tokenizer, cfg.text_field,
            max_samples, cfg.max_seq_len, cfg.seed,
            batch_size=1,
        ):
            ids_t  = torch.tensor(batch.input_ids,      dtype=torch.long, device=device)
            mask_t = torch.tensor(batch.attention_mask,  dtype=torch.long, device=device)
            seq_len = ids_t.shape[1]
            _ = model(input_ids=ids_t, attention_mask=mask_t, use_cache=False)
            traces  = tracer.pop_last_forward(seq_len)
            n_batches += 1

            for l in range(n_pairs):
                src = traces[l]      # layer l
                tgt = traces[l + 1]  # layer l+1

                if tgt.num_experts == 0 or tgt.true_topk_scores is None:
                    continue

                ids   = tgt.true_topk_ids.numpy()    # (T, K)
                scrs  = tgt.true_topk_scores.numpy() # (T, K)

                W_gate = tracer.get_router_weight_EH(l + 1).to(device)
                ln_A   = tracer.get_input_layernorm(l + 1)       # input LN (wrong for gate)
                ln_B   = tracer.get_post_attn_layernorm(l + 1)   # post-attn LN (correct)

                # ── Signal A (baseline) ─────────────────────────────────────
                p_A = _gate_probs(src.post_attn_states, W_gate, ln_A)  # (T, E)

                # ── Signal B (post-MoE + correct LN) ───────────────────────
                h_tgt = tgt.layer_input_states   # h_in_{l+1}  (T, H) cpu
                if h_tgt is not None:
                    p_B = _gate_probs(h_tgt, W_gate, ln_B)
                else:
                    p_B = p_A   # fallback (should not occur for Qwen)

                # ── Signal D (velocity extrapolation) — logits for E ──────
                h_src = src.layer_input_states   # h_in_l  (T, H) cpu
                if h_src is not None and h_tgt is not None:
                    h_extrap = (2.0 * h_tgt.float() - h_src.float()).to(h_tgt.dtype)
                    z_D = _gate_logits(h_extrap, W_gate, ln_B)  # (T, E) raw logits
                    p_D = _softmax_numpy(z_D)
                    del h_extrap
                else:
                    z_D = None
                    p_D = p_B

                # ── Signal BD (prob-space blend B+D, kept for reference) ────
                p_BD = bd_alpha * p_B + (1.0 - bd_alpha) * p_D
                tot  = p_BD.sum(axis=1, keepdims=True)
                p_BD /= np.where(tot > 0, tot, 1.0)

                # ── Signal C (causal EMA from B) ────────────────────────────
                p_C = _compute_ema_prior(p_B, K, beta=ema_beta, lam=ema_lambda)

                # ── Compute logit_B for E sweep ──────────────────────────────
                z_B = _gate_logits(h_tgt, W_gate, ln_B) if h_tgt is not None else None

                # ── Signals A/B/C/D/BD: precompute → stream ─────────────────
                for sig_idx, p in enumerate([p_A, p_B, p_C, p_D, p_BD]):
                    cp, srank, ccov = _precompute_sorted(p, ids, scrs)
                    if mode == "cal":
                        stats[sig_idx].update_cal(l, cp, srank, ccov, gamma_grid)
                    else:
                        stats[sig_idx].update_tst(l, cp, srank, ccov)
                    del cp, srank, ccov

                del p_A, p_C, p_BD

                # ── Signal E: logit-space blend sweep ───────────────────────
                # z_ε = (1−ε)·z_B + ε·z_D  →  softmax
                # Cal: sweep all ε values; Tst: apply best_eps_per_layer[l]
                if z_B is not None and z_D is not None:
                    if mode == "cal" and eps_grid is not None and eps_stats is not None:
                        for ei, eps in enumerate(eps_grid):
                            z_e = (1.0 - eps) * z_B + eps * z_D  # (T, E)
                            p_e = _softmax_numpy(z_e)
                            cp, srank, ccov = _precompute_sorted(p_e, ids, scrs)
                            eps_stats[ei].update_cal(l, cp, srank, ccov, gamma_grid)
                            del cp, srank, ccov, p_e, z_e
                    elif mode == "tst" and best_eps_per_layer is not None and sig_E_stats is not None:
                        eps = best_eps_per_layer[l]
                        z_e = (1.0 - eps) * z_B + eps * z_D
                        p_e = _softmax_numpy(z_e)
                        cp, srank, ccov = _precompute_sorted(p_e, ids, scrs)
                        sig_E_stats.update_tst(l, cp, srank, ccov)
                        del cp, srank, ccov, p_e, z_e

                del p_B, p_D, z_B, z_D

            if n_batches % 50 == 0:
                total = n_batches * seq_len * n_pairs
                print(f"  [{desc}] batch={n_batches}  records={total}", flush=True)


# ═══════════════════════════════════════════════════════════ main entry point

def run_qwen_signals(
    cfg:        ExperimentConfig,
    n_gamma:    int   = 400,
    ema_beta:   float = 0.30,
    ema_lambda: float = 0.30,
    bd_alpha:   float = 0.70,
    eps_grid:   Optional[np.ndarray] = None,   # ε values for logit-blend Signal E
                                                # default: 9 values in [0.05, 0.50]
) -> dict:
    """
    Compare 5 + 1 predictor signals under the identical CRC framework.
    Signal E (velocity-corrected, per-layer optimal ε) is the novel contribution:
      z_ε(l) = (1−ε_l)·z_B + ε_l·z_D  →  softmax
    ε_l is selected per layer to maximise calibration rank-4 hit rate.
    Streaming implementation: O(1) extra RAM w.r.t. model size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    alpha  = cfg.alpha
    K      = cfg.router_top_k

    # Default ε sweep: 9 values covering 0 (=B) to 0.5
    if eps_grid is None:
        eps_grid = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                            dtype=np.float32)
    n_eps = len(eps_grid)

    MODEL_PATH = (
        "/models/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/"
        "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    )

    # ── 1. Load tokenizer + model ──────────────────────────────────────────
    print(f"[sig] Loading tokenizer from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    print(f"[sig] Loading model → dtype={cfg.dtype}, device={device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype, device_map=str(device), trust_remote_code=False
    )
    model.eval()
    print(
        f"[sig] Model loaded.  Params: "
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B",
        flush=True,
    )

    # ── 2. Data ────────────────────────────────────────────────────────────
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
    print(f"[sig] Hooked {n_layers} decoder layers.  {n_pairs} cross-layer pairs.", flush=True)
    print(
        f"[sig] Signal hyper-params: ema_beta={ema_beta}  ema_lambda={ema_lambda}"
        f"  bd_alpha={bd_alpha}  n_gamma={n_gamma}"
        f"  eps_grid={eps_grid.tolist()}",
        flush=True,
    )

    gamma_grid = np.linspace(0.01, 0.9999, n_gamma)
    n_sigs = len(_SIG_NAMES)
    stats      = [_StreamStats(n_pairs, n_gamma, K) for _ in range(n_sigs)]
    # One _StreamStats per ε value for Signal E calibration
    eps_stats  = [_StreamStats(n_pairs, n_gamma, K) for _ in range(n_eps)]

    # ── 4. Calibration pass (streaming) ───────────────────────────────────
    print("\n[sig] === Calibration pass (streaming — O(1) RAM) ===", flush=True)
    t0 = time.time()
    _stream_pass(
        tracer, tokenizer, model, device, cal_data, cfg,
        cfg.max_samples_cal, "cal", stats, gamma_grid,
        mode="cal", ema_beta=ema_beta, ema_lambda=ema_lambda, bd_alpha=bd_alpha,
        eps_grid=eps_grid, eps_stats=eps_stats,
    )
    print(f"[sig] Calibration pass done in {time.time()-t0:.0f}s", flush=True)

    # Report approximate n_cal tokens
    n_cal_tok = stats[0].cal_n_tok[next(
        l for l in range(n_pairs) if stats[0].cal_n_tok[l] > 0
    )]
    print(f"[sig] n_cal tokens per layer ≈ {n_cal_tok}", flush=True)

    # ── 5. Calibrate γ* per signal per layer ──────────────────────────────
    for s in range(n_sigs):
        stats[s].calibrate(alpha, n_cal_tok, gamma_grid)
    for ei in range(n_eps):
        n_ei = eps_stats[ei].cal_n_tok[next(
            (l for l in range(n_pairs) if eps_stats[ei].cal_n_tok[l] > 0), 0
        )]
        eps_stats[ei].calibrate(alpha, n_ei if n_ei > 0 else n_cal_tok, gamma_grid)

    # ── 6. Find best ε per layer (by calibration rank-4 hit rate) ─────────
    # cal_rh_sum[l][4] / cal_n_tok[l]  for each ε, pick argmax
    best_eps_per_layer: List[float] = []
    best_eps_r4_per_layer: List[float] = []
    print("\n[sig] Signal E — per-layer optimal ε (by cal rank-4 hit rate):")
    print(f"  {'Layer':<8}" + "".join(f"  {eps:.2f}" for eps in eps_grid) + "  best_ε")
    for l in range(n_pairs):
        r4_by_eps = []
        for ei in range(n_eps):
            n_l = eps_stats[ei].cal_n_tok[l]
            if n_l > 0:
                r4 = float(eps_stats[ei].cal_rh_sum[l][4]) / n_l
            else:
                r4 = 0.0
            r4_by_eps.append(r4)
        best_ei   = int(np.argmax(r4_by_eps))
        best_eps  = float(eps_grid[best_ei])
        best_r4   = r4_by_eps[best_ei]
        best_eps_per_layer.append(best_eps)
        best_eps_r4_per_layer.append(best_r4)
        row = f"  {l:02d}→{l+1:02d}  "
        row += "".join(f"  {r4:.4f}" for r4 in r4_by_eps)
        row += f"  ε={best_eps:.2f}"
        print(row)

    # Signal E inherits γ* from the winning ε's stats per layer
    sig_E_stats = _StreamStats(n_pairs, n_gamma, K)
    sig_E_stats.gamma_stars = [
        eps_stats[int(np.argmax([
            float(eps_stats[ei].cal_rh_sum[l][4]) / max(eps_stats[ei].cal_n_tok[l], 1)
            for ei in range(n_eps)
        ]))].gamma_stars[l]
        for l in range(n_pairs)
    ]

    print(f"\n[sig] Best ε per layer: {best_eps_per_layer}", flush=True)

    # Print γ* comparison table for A/B/D/E
    print("\n[sig] Per-layer γ* by signal:")
    print(f"  {'Layer':<8}" + "".join(f"  {'γ*-'+nm:>8}" for nm in _SIG_NAMES))
    for l in range(n_pairs):
        row = f"  {l:02d}→{l+1:02d}  "
        row += "".join(f"  {stats[s].gamma_stars[l]:8.4f}" for s in range(n_sigs))
        print(row)

    # Calibration rank metrics
    print("\n[sig] Calibration rank-4 hit rates:")
    for s, nm in enumerate(_SIG_NAMES):
        n_c = sum(stats[s].cal_n_tok)
        if n_c > 0:
            r4 = sum(stats[s].cal_rh_sum[l][4] for l in range(n_pairs)) / n_c
            print(f"  [{nm}] cal rank-4 hit rate = {r4:.4f}")
    # Signal E: use best_eps_r4_per_layer averaged
    r4_E_cal = float(np.mean([best_eps_r4_per_layer[l] for l in range(n_pairs)
                               if n_cal_tok > 0]))
    print(f"  [E ] cal rank-4 hit rate = {r4_E_cal:.4f}  (per-layer best ε)")

    # ── 7. Test pass (streaming) ───────────────────────────────────────────
    print("\n[sig] === Test pass (streaming) ===", flush=True)
    t0 = time.time()
    _stream_pass(
        tracer, tokenizer, model, device, test_data, cfg,
        cfg.max_samples_test, "test", stats, gamma_grid,
        mode="tst", ema_beta=ema_beta, ema_lambda=ema_lambda, bd_alpha=bd_alpha,
        best_eps_per_layer=best_eps_per_layer, sig_E_stats=sig_E_stats,
    )
    print(f"[sig] Test pass done in {time.time()-t0:.0f}s", flush=True)

    # ── 8. Collect results ─────────────────────────────────────────────────
    all_results = {nm: stats[s].results(K) for s, nm in enumerate(_SIG_NAMES)}
    all_results["E"] = sig_E_stats.results(K)

    _DESC = {
        "A":  "A: baseline  (h_post_attn + in_LN)",
        "B":  "B: post-MoE  (h_in + pa_LN)",
        "C":  "C: B + causal EMA momentum",
        "D":  "D: velocity extrapolation",
        "BD": "BD: 0.70·B + 0.30·D (prob-space)",
        "E":  "E: vel-corrected logit blend (novel)",
    }

    # ── 9. Print detailed metrics for every signal ─────────────────────────
    for nm in list(_SIG_NAMES) + ["E"]:
        m  = all_results[nm]
        rh = m.get("rank_hit", {})
        cv = m.get("cov_at",   {})
        print(f"\n  ══ {_DESC[nm]} ══")
        print(f"  Adp avg|S|={m.get('avg_set',0):.2f}  E[L]={m.get('mean_L',0):.4f}  "
              f"P(full)={m.get('p_full',0):.4f}")
        print(f"  Rank hits:  r1={rh.get(1,0):.4f}  r2={rh.get(2,0):.4f}  "
              f"r3={rh.get(3,0):.4f}  r4={rh.get(4,0):.4f}")
        print(f"  Cov@k  :  c@1={cv.get(1,0):.4f}  c@2={cv.get(2,0):.4f}  "
              f"c@3={cv.get(3,0):.4f}  c@4={cv.get(4,0):.4f}")

    # ── 10. Summary table ──────────────────────────────────────────────────
    SEP = "═" * 114
    print(f"\n{SEP}")
    print(
        f"[sig] SIGNAL COMPARISON  "
        f"(α={alpha}  K={K}  E=60  n_cal≈{n_cal_tok}  "
        f"β={ema_beta}  λ={ema_lambda}  bd_α={bd_alpha})"
    )
    print(SEP)
    hdr = (
        f"  {'Signal':<42}  {'Adp|S|':>7}  {'Adp E[L]':>9}  "
        f"{'P(full)':>8}  {'r1':>6}  {'r2':>6}  {'r3':>6}  {'r4':>6}  "
        f"{'c@1':>6}  {'c@2':>6}  {'c@3':>6}  {'c@4':>6}"
    )
    print(hdr)
    print("  " + "─" * 110)
    for nm in list(_SIG_NAMES) + ["E"]:
        m  = all_results[nm]
        rh = m.get("rank_hit", {})
        cv = m.get("cov_at",   {})
        print(
            f"  {_DESC[nm]:<36}  "
            f"{m.get('avg_set',0):7.2f}  "
            f"{m.get('mean_L',0):9.4f}  "
            f"{m.get('p_full',0):8.4f}  "
            f"{rh.get(1,0):6.4f}  {rh.get(2,0):6.4f}  "
            f"{rh.get(3,0):6.4f}  {rh.get(4,0):6.4f}  "
            f"{cv.get(1,0):6.4f}  {cv.get(2,0):6.4f}  "
            f"{cv.get(3,0):6.4f}  {cv.get(4,0):6.4f}"
        )
    print(SEP)
    print(
        "\n  Key:\n"
        "  Signal A : BASELINE  h_post_attn_l  + input_LN_{l+1}  (wrong h AND wrong LN)\n"
        "  Signal B : POST-MoE  h_in_{l+1}     + post_attn_LN_{l+1}  (correct both)\n"
        "  Signal C : B + CAUSAL EMA PRIOR  (routing locality / intra-sequence coherence)\n"
        f"             β={ema_beta}  λ={ema_lambda}\n"
        "  Signal D : VELOCITY  h_extrap = 2·h_in_{l+1} - h_in_l  (residual momentum)\n"
        f"  Signal BD: BLEND  {bd_alpha:.2f}·B + {1-bd_alpha:.2f}·D  (prob-space)\n"
        "  Signal E : VEL-CORRECTED LOGIT BLEND  z_ε=(1-ε)·z_B + ε·z_D → softmax\n"
        "             per-layer ε selected on calibration by rank-4 hit rate (NOVEL)\n"
        "\n"
        "  Adp|S|   = mean set size under adaptive CRC  (CRC guarantee: E[L] ≤ α)\n"
        "  r-k      = P(expert at predicted rank k is one of the true K)\n"
        "  c@k      = P(all top-k predicted experts are true experts)\n"
        "  Streaming: probs discarded after each sequence — peak extra RAM ≈ 15 MB\n"
    )

    # Print per-layer best-eps selection
    print("[sig] Per-layer best-ε for Signal E:")
    for l, eps_val in enumerate(best_eps_per_layer):
        print(f"  layer {l:02d}→{l+1:02d}  ε={eps_val:.2f}")

    # Per-layer γ* in results
    all_sig_keys = list(_SIG_NAMES) + ["E"]
    gamma_table = {
        nm: all_results[nm].get("gamma_stars", []) for nm in all_sig_keys
    }
    print("[sig] Per-layer γ* table:")
    print(f"  {'Layer':<8}" + "".join(f"  {'γ*-'+nm:>8}" for nm in all_sig_keys))
    for l in range(n_pairs):
        row = f"  {l:02d}→{l+1:02d}  "
        row += "".join(
            f"  {gamma_table[nm][l]:8.4f}" if l < len(gamma_table[nm]) else "       N/A"
            for nm in all_sig_keys
        )
        print(row)

    tracer.detach()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "alpha":               alpha,
        "ema_beta":            ema_beta,
        "ema_lambda":          ema_lambda,
        "bd_alpha":            bd_alpha,
        "n_gamma":             n_gamma,
        "best_eps_per_layer":  best_eps_per_layer,
        "eps_grid":            eps_grid,
        **{f"metrics_{nm}": all_results[nm] for nm in list(_SIG_NAMES) + ["E"]},
    }
