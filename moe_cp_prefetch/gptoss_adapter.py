"""
gptoss_adapter.py — Hooks + router-param extraction for openai/gpt-oss-20b
===========================================================================

GptOssDecoderLayer forward (relevant excerpt):
  residual   = hidden_states                              # ← h_in_l  (B, T, H)
  hidden_states = input_layernorm(hidden_states)
  hidden_states, _ = self_attn(hidden_states, ...)
  hidden_states = residual + hidden_states                # ← h_post_attn_l

  residual = hidden_states
  hidden_states = post_attention_layernorm(hidden_states) # ← LN of h_post_attn_l
  hidden_states, _ = mlp(hidden_states)                   # ← router sees this
  hidden_states = residual + hidden_states                # ← h_in_{l+1}  (B, T, H)

GptOssMLP.forward receives (B, T, H) and reshapes to (T, H) *before* the router:
  hidden_states = hidden_states.reshape(-1, hidden_dim)   # (T, H)
  _, router_scores, router_indices = self.router(hidden_states)

GptOssTopKRouter.forward:
  router_logits  = F.linear(h, weight, bias)              (T, E)  ← raw logits
  top_val, top_idx = topk(router_logits, K, dim=-1)       (T, K)
  router_scores  = softmax(top_val, dim=1)                (T, K)  ← NOT full softmax
  return router_logits, router_scores, router_indices

Module path to decoder layers:
  GptOssForCausalLM
    .model                → GptOssModel
      .layers[l]          → GptOssDecoderLayer
        .mlp              → GptOssMLP
          .router         → GptOssTopKRouter

Key constants for gpt-oss-20b:
  hidden_size    H = 2880
  num_local_experts  E = 32
  num_experts_per_tok  K = 4
  num_hidden_layers  L = 24   (ALL 24 layers have MoE)
  weights: router is bf16; experts are mxfp4 (auto-dequantized to bf16 if kernels not installed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

MODEL_PATH = (
    "/models/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee"
)


# ════════════════════════════════════════════════ Router params (numpy, CPU) ══

@dataclass
class GptOssRouterParams:
    """Cached router parameters for one layer, as float32 numpy arrays."""
    W_proj: np.ndarray    # (E, H) — router.weight
    b_proj: np.ndarray    # (E,)   — router.bias


def extract_router_params(model) -> List[GptOssRouterParams]:
    """
    Extract GptOssTopKRouter weights for all decoder layers as float32 CPU numpy.

    Args:
        model: GptOssForCausalLM (or GptOssModel) — any obj with .model.layers
               or .layers attribute.

    Returns:
        List of length n_layers; each entry is a GptOssRouterParams.
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    params: List[GptOssRouterParams] = []
    for layer in layers:
        router = layer.mlp.router
        params.append(GptOssRouterParams(
            W_proj = router.weight.detach().float().cpu().numpy(),   # (E, H)
            b_proj = router.bias.detach().float().cpu().numpy(),     # (E,)
        ))
    return params


def router_logits_np(h: np.ndarray, rp: GptOssRouterParams) -> np.ndarray:
    """
    Replicate GptOssTopKRouter gate computation in pure numpy.

    GptOssTopKRouter.forward does F.linear(h, weight, bias), i.e.:
        logits = h @ weight.T + bias

    The router receives h = post_attention_layernorm(h_post_attn), already
    reshaped to (T, H), so for Signal B (exact) we apply this formula directly
    to the captured pre-hook input.

    For Signal A (early) we apply the same weight matrix to the previous
    layer's output (h_in_{l-1}, raw un-normalised residual stream) — it is
    an approximate cross-layer predictor, intentionally so.

    Args:
        h   (T, H) float32 numpy
        rp  GptOssRouterParams for the target layer

    Returns:
        logits  (T, E) float32 numpy — raw router logits (before softmax/topk)
    """
    return (h @ rp.W_proj.T + rp.b_proj).astype(np.float32)


# ════════════════════════════════════════════════════════ Dual-signal tracer ══

class GptOssDualTracer:
    """
    Registers forward hooks on every GptOssDecoderLayer and its router to
    capture the three signals needed for the dual-signal (E_A + E_B) pipeline:

      h_in_list[l]       — output of decoder layer l, shape (T, H) float32 numpy.
                           Used as Signal-A input for layer l+1:
                             z_A[l+1] = router_logits_np(h_in_list[l], rp[l+1])
                           Also used for the cross-sequence carry at layer 0.

      h_pre_router[l]    — input to layer l's router (= post_attention_layernorm
                           applied to h_post_attn_l, already reshaped to (T, H)
                           by GptOssMLP.forward before calling router).
                           Used as Signal-B input:
                             z_B[l] = router_logits_np(h_pre_router[l], rp[l])

      true_topk_ids[l]   — top-K expert indices, shape (T, K) int32 numpy.
                           Ground truth for hit-rate and CRC metrics.

      true_topk_scrs[l]  — softmax(top-K logits), shape (T, K) float32 numpy.
                           Used as true routing weights in CRC loss calculation.

    Note: all numpy arrays are on CPU float32, safe to use across multiple GPU
    shards set up by device_map='auto'.
    """

    def __init__(self, model):
        layers = model.model.layers if hasattr(model, "model") else model.layers
        n = len(layers)
        self.n_layers = n

        self.h_in_list:     List[Optional[np.ndarray]] = [None] * n
        self.h_pre_router:  List[Optional[np.ndarray]] = [None] * n
        self.true_topk_ids: List[Optional[np.ndarray]] = [None] * n
        self.true_topk_scrs:List[Optional[np.ndarray]] = [None] * n
        self._hooks: list = []

        for l, layer in enumerate(layers):
            # ── Layer output hook: captures h_in_{l+1} = output of layer l ──
            def _layer_outhook(mod, inp, out, _l=l):
                h = out[0] if isinstance(out, tuple) else out
                arr = h.detach().float().cpu().numpy()
                if arr.ndim == 3:          # (B, T, H) with B=1 → (T, H)
                    arr = arr[0]
                self.h_in_list[_l] = arr
            self._hooks.append(layer.register_forward_hook(_layer_outhook))

            # ── Router pre-hook: captures LN(h_post_attn) flattened (T, H) ──
            def _router_prehook(mod, inp, _l=l):
                # inp[0] is hidden_states reshaped to (T, H) by GptOssMLP
                self.h_pre_router[_l] = inp[0].detach().float().cpu().numpy()
            self._hooks.append(
                layer.mlp.router.register_forward_pre_hook(_router_prehook)
            )

            # ── Router output hook: captures top-K indices + scores ──────────
            def _router_outhook(mod, inp, out, _l=l):
                # out: (router_logits (T,E), router_scores (T,K), router_indices (T,K))
                self.true_topk_scrs[_l] = out[1].detach().float().cpu().numpy()
                self.true_topk_ids[_l]  = out[2].detach().cpu().numpy().astype(np.int32)
            self._hooks.append(
                layer.mlp.router.register_forward_hook(_router_outhook)
            )

    def reset(self) -> None:
        """Clear all captured numpy arrays (call after processing each sequence)."""
        for i in range(self.n_layers):
            self.h_in_list[i]      = None
            self.h_pre_router[i]   = None
            self.true_topk_ids[i]  = None
            self.true_topk_scrs[i] = None

    def remove(self) -> None:
        """Remove all registered hooks (call when done with all passes)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
