"""
qwen35_adapter.py — Hooks + router-param extraction for Qwen/Qwen3.5-35B-A3B
==============================================================================

Qwen3_5MoeDecoderLayer forward (simplified):
  residual   = hidden_states                                  # ← h_in_l  (B, T, H)
  hidden_states = self.input_layernorm(hidden_states)
  # Token mixer: linear_attention (layers 0,1,2,4,5,6,…) or full_attention (3,7,11,…)
  hidden_states = residual + hidden_states                    # ← h_post_attn_l

  residual = hidden_states
  hidden_states = self.post_attention_layernorm(hidden_states)
  hidden_states = self.mlp(hidden_states)                     # ← SparseMoeBlock
  hidden_states = residual + hidden_states                    # ← h_out = h_in_{l+1}

Qwen3_5MoeSparseMoeBlock.forward receives (B, T, H) and reshapes BEFORE calling gate:
  hidden_states_reshaped = hidden_states.view(-1, hidden_dim)  # (T, H) for B=1
  _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

Qwen3_5MoeTopKRouter.forward:
  hidden_states = hidden_states.reshape(-1, hidden_dim)        # (T, H)
  router_logits = F.linear(hidden_states, self.weight)         # (T, E)  raw, NO bias
  router_logits = softmax(router_logits, dim=-1)               # (T, E)  full softmax
  router_top_value, router_indices = topk(router_logits, K, dim=-1)
  router_top_value /= router_top_value.sum(dim=-1, keepdim=True)  # re-normalise
  return router_logits, router_top_value, router_indices       # (T,E), (T,K), (T,K)

Module path to decoder layers (two possibilities, both handled):
  Qwen3_5MoeForConditionalGeneration (architecture in config.json):
    .model                   → Qwen3_5MoeModel
      .language_model        → Qwen3_5MoeTextModel
        .layers[l]           → Qwen3_5MoeDecoderLayer
          .mlp               → Qwen3_5MoeSparseMoeBlock
            .gate            → Qwen3_5MoeTopKRouter
  Qwen3_5MoeForCausalLM (AutoModelForCausalLM fallback via model_type):
    .model                   → Qwen3_5MoeTextModel
      .layers[l]             → Qwen3_5MoeDecoderLayer   (same from here)

Key constants for Qwen3.5-35B-A3B  (text_config):
  hidden_size           H = 2048
  num_experts           E = 256
  num_experts_per_tok   K = 8
  num_hidden_layers     L = 40   (ALL 40 layers are MoE; mlp_only_layers=[])
  Router weight:  (E=256, H=2048) bf16, NO bias
  ~35B params bf16 ≈ 70 GB → needs 2 A6000 GPUs (48 GB each)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

MODEL_PATH = (
    "/models/models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)


# ════════════════════════════════════════════════ Router params (numpy, CPU) ══

@dataclass
class Qwen35RouterParams:
    """Cached router parameters for one layer, as float32 numpy arrays."""
    W_proj: np.ndarray    # (E, H) — gate.weight, NO bias


def _get_decoder_layers(model):
    """
    Return the nn.ModuleList of Qwen3_5MoeDecoderLayer objects regardless of
    whether AutoModelForCausalLM loaded the text-only or multimodal variant.

    Paths tried in order:
      1. model.model.language_model.layers   (Qwen3_5MoeForConditionalGeneration)
      2. model.model.layers                  (Qwen3_5MoeForCausalLM)
    """
    text = getattr(model, "model", None)
    if text is None:
        raise RuntimeError("Model has no .model attribute.")

    lm = getattr(text, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers

    if hasattr(text, "layers"):
        return text.layers

    raise RuntimeError(
        "Cannot find decoder layers: tried .model.language_model.layers "
        "and .model.layers — neither exists."
    )


def extract_router_params(model) -> List[Qwen35RouterParams]:
    """
    Extract Qwen3_5MoeTopKRouter weights for all decoder layers as float32 CPU numpy.

    Args:
        model: Qwen3_5MoeForCausalLM or Qwen3_5MoeForConditionalGeneration.

    Returns:
        List of length n_layers; each entry is a Qwen35RouterParams.
    """
    layers = _get_decoder_layers(model)
    params: List[Qwen35RouterParams] = []
    for layer in layers:
        gate = layer.mlp.gate   # Qwen3_5MoeTopKRouter
        params.append(Qwen35RouterParams(
            W_proj=gate.weight.detach().float().cpu().numpy(),  # (E, H)
        ))
    return params


def router_logits_np(h: np.ndarray, rp: Qwen35RouterParams) -> np.ndarray:
    """
    Replicate Qwen3_5MoeTopKRouter gate computation in pure numpy (pre-softmax).

    Qwen3_5MoeTopKRouter.forward computes F.linear(h, weight) — NO bias:
        raw_logits = h @ weight.T       (T, E)

    The router then applies softmax(raw_logits, dim=-1) before topk.
    We return RAW logits here; the pipeline calls _softmax_numpy afterward
    (consistent: softmax is monotonic so topk of raw == topk of softmax).

    For Signal B (exact): h is captured at the gate pre-hook, i.e.
        post_attention_layernorm(h_post_attn), already flattened to (T, H)
        by Qwen3_5MoeSparseMoeBlock.forward before calling gate.

    For Signal A (early): h is the un-normalised residual output of layer l-1
        (intentional approximation; fires before layer l's attention runs).

    Args:
        h   (T, H) float32 numpy
        rp  Qwen35RouterParams for the target layer

    Returns:
        raw_logits  (T, E) float32 numpy — before softmax/topk
    """
    return (h @ rp.W_proj.T).astype(np.float32)


# ════════════════════════════════════════════════════════ Dual-signal tracer ══

class Qwen35DualTracer:
    """
    Registers forward hooks on every Qwen3_5MoeDecoderLayer and its gate to
    capture the three signals needed for the dual-signal (E_A + E_B) pipeline:

      h_in_list[l]       — output of decoder layer l, shape (T, H) float32 numpy.
                           Used as Signal-A input for layer l+1:
                             z_A[l+1] = router_logits_np(h_in_list[l], rp[l+1])
                           Also used for cross-sequence carry at layer 0.

      h_pre_router[l]    — input to layer l's gate (= post_attention_layernorm
                           applied to h_post_attn_l, already reshaped to (T, H)
                           by Qwen3_5MoeSparseMoeBlock before calling gate).
                           Used as Signal-B input:
                             z_B[l] = router_logits_np(h_pre_router[l], rp[l])

      true_topk_ids[l]   — top-K expert indices, shape (T, K) int32 numpy.
                           Ground truth for hit-rate and CRC metrics.

      true_topk_scrs[l]  — re-normalised top-K gate weights, shape (T, K) float32.
                           Used as true routing weights in CRC loss calculation.

    All arrays are CPU float32, safe across multi-GPU device_map='auto' shards.
    """

    def __init__(self, model):
        layers = _get_decoder_layers(model)
        n = len(layers)
        self.n_layers = n

        self.h_in_list:      List[Optional[np.ndarray]] = [None] * n
        self.h_pre_router:   List[Optional[np.ndarray]] = [None] * n
        self.true_topk_ids:  List[Optional[np.ndarray]] = [None] * n
        self.true_topk_scrs: List[Optional[np.ndarray]] = [None] * n
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

            # ── Gate pre-hook: captures LN(h_post_attn) already flat (T, H) ─
            # Qwen3_5MoeSparseMoeBlock reshapes (B,T,H)→(T,H) before calling gate.
            def _gate_prehook(mod, inp, _l=l):
                self.h_pre_router[_l] = inp[0].detach().float().cpu().numpy()
            self._hooks.append(
                layer.mlp.gate.register_forward_pre_hook(_gate_prehook)
            )

            # ── Gate output hook: captures top-K indices + re-normalised scores ─
            def _gate_outhook(mod, inp, out, _l=l):
                # out: (router_logits (T,E), router_scores (T,K), router_indices (T,K))
                # router_scores: re-normalised top-K weights (sum to 1 per token)
                self.true_topk_scrs[_l] = out[1].detach().float().cpu().numpy()
                self.true_topk_ids[_l]  = out[2].detach().cpu().numpy().astype(np.int32)
            self._hooks.append(
                layer.mlp.gate.register_forward_hook(_gate_outhook)
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
