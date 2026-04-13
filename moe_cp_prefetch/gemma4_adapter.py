"""
gemma4_adapter.py — Hooks + router-param extraction for Gemma4ForConditionalGeneration
========================================================================================

Gemma4TextDecoderLayer forward (relevant excerpt):
  residual = hidden_states                              # ← h_in_l
  hidden_states = input_layernorm(hidden_states)
  hidden_states = self_attn(hidden_states, ...)
  hidden_states = post_attention_layernorm(hidden_states)
  hidden_states = residual + hidden_states              # ← h_post_attn_l
  residual = hidden_states                              # start of FFN/MoE block

  hidden_states = pre_feedforward_layernorm(hidden_states)
  hidden_states = mlp(hidden_states)                   # shared MLP (runs in parallel with MoE)

  if enable_moe_block:
      hidden_states_flat = residual.reshape(-1, H)      # = h_post_attn_l flattened
      _, top_k_weights, top_k_index = self.router(hidden_states_flat)  # ← Signal-B input

  hidden_states = post_feedforward_layernorm(combined)
  hidden_states = residual + hidden_states              # ← layer output = h_in_{l+1}

Module path to decoder layers:
  Gemma4ForConditionalGeneration
    .model                  → Gemma4Model
      .language_model       → Gemma4TextModel
        .layers[l]          → Gemma4TextDecoderLayer
          .router           → Gemma4TextRouter  (present on ALL 30 layers, K=8, E=128)

Gemma4TextRouter.forward internals (used to reconstruct logits in numpy):
  norm(h)                   RMSNorm WITHOUT learnable scale (with_scale=False)
  * self.scale              learnable per-dim scale  (H,)
  * self.scalar_root_size   = H^{-0.5}  (Python float)
  @ self.proj.weight.T      Linear H → E, no bias
  → softmax → top-8

Key model constants for gemma4-26B-A4B:
  hidden_size   H = 2816
  num_experts   E = 128
  top_k_experts K = 8
  num_layers    L = 30   (ALL 30 layers are MoE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

MODEL_PATH = (
    "/models/models--google--gemma-4-26B-A4B/snapshots/"
    "64143b04706fadeb2f8ac198f7ecab57b94b1e0b"
)


# ═══════════════════════════════════════════ Router params (numpy, on CPU) ══

@dataclass
class Gemma4RouterParams:
    """Cached router parameters for one layer, kept as float32 numpy arrays."""
    W_proj:      np.ndarray   # (E, H) — router.proj.weight
    W_scale:     np.ndarray   # (H,)   — router.scale
    scalar_root: float        # H^{-0.5} = router.scalar_root_size
    eps:         float        # RMSNorm epsilon


def extract_router_params(text_model) -> List[Optional[Gemma4RouterParams]]:
    """
    Walk all decoder layers and extract router parameters as float32 numpy
    arrays (GPU memory freed).  Returns a list of length n_layers; entry is
    None for any layer that somehow lacks a router.
    """
    params: List[Optional[Gemma4RouterParams]] = []
    for layer in text_model.layers:
        if not hasattr(layer, "router"):
            params.append(None)
            continue
        router = layer.router
        params.append(Gemma4RouterParams(
            W_proj      = router.proj.weight.detach().float().cpu().numpy(),
            W_scale     = router.scale.detach().float().cpu().numpy(),
            scalar_root = float(router.scalar_root_size),
            eps         = float(router.eps),
        ))
    return params


def router_logits_np(h: np.ndarray, rp: Gemma4RouterParams) -> np.ndarray:
    """
    Replicate Gemma4TextRouter's gate computation entirely in numpy (no GPU).

    Steps (mirror of router.forward):
      1. RMSNorm without learnable scale: h /= sqrt(mean(h^2) + eps)
      2. Multiply by learnable scale * scalar_root_size
      3. Linear projection: @ W_proj.T   (→ raw logits before softmax)
      4. Return raw logits — caller applies softmax for CRC / rank metrics.

    Args:
        h  (T, H) float32 numpy — hidden states (any batch dimension flattened)
        rp           Gemma4RouterParams for this layer

    Returns:
        z  (T, E) float32 numpy — raw router logits
    """
    # Step 1 — RMSNorm (no learnable scale in router.norm)
    sq_mean = np.mean(h * h, axis=-1, keepdims=True)          # (T, 1)
    h_normed = h / np.sqrt(sq_mean + rp.eps)                  # (T, H)
    # Step 2 — learnable per-dim scale + fixed scalar
    h_scaled = h_normed * (rp.W_scale * rp.scalar_root)       # (T, H)
    # Step 3 — linear projection (E, H) × (H, T) → (T, E)
    logits = h_scaled @ rp.W_proj.T                           # (T, E)
    return logits.astype(np.float32)


# ═══════════════════════════════════════════════════════ Dual-signal tracer ══

class Gemma4DualTracer:
    """
    Registers PyTorch hooks on every decoder layer and every router, capturing:

      h_in_list[l]          float32 numpy (T, H)
          = output of decoder layer l  (= h_in_{l+1}, used for E_A of layer l+1)

      h_post_attn_list[l]   float32 numpy (T, H)
          = direct input to router of layer l  (= h_post_attn_l, used for E_B of l)

      true_topk_ids[l]      int32 numpy (T, K)
          = expert indices actually selected by router of layer l

      true_topk_scrs[l]     float32 numpy (T, K)
          = normalised weights for the selected experts (used to compute coverage)

    All tensors are moved to CPU as float32 immediately in the hook to avoid
    accumulating per-GPU VRAM across a long sequence.  Works with device_map="auto"
    (tensors may originate from any device).

    Usage:
        tracer = Gemma4DualTracer(model.model.language_model)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        # tracer.*_list[l]  are populated for all layers
        tracer.reset()   # free numpy arrays between sequences
        tracer.remove()  # when done with all sequences
    """

    def __init__(self, text_model):
        layers = text_model.layers
        self.n_layers         = len(layers)
        self.h_in_list        = [None] * self.n_layers
        self.h_post_attn_list = [None] * self.n_layers
        self.true_topk_ids    = [None] * self.n_layers
        self.true_topk_scrs   = [None] * self.n_layers
        self._hooks           = []

        for l, layer in enumerate(layers):
            # ── layer forward hook  → captures h_in_{l+1}
            self._hooks.append(
                layer.register_forward_hook(self._layer_hook(l))
            )
            if hasattr(layer, "router"):
                # ── router pre-hook → captures h_post_attn_l (router input)
                self._hooks.append(
                    layer.router.register_forward_pre_hook(self._router_prehook(l))
                )
                # ── router forward hook → captures true routing decisions
                self._hooks.append(
                    layer.router.register_forward_hook(self._router_outhook(l))
                )

    # ─── hook factories ───────────────────────────────────────────────────

    def _layer_hook(self, l: int):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            if isinstance(t, torch.Tensor):
                self.h_in_list[l] = (
                    t.detach().float()
                     .reshape(-1, t.shape[-1])
                     .cpu().numpy()
                )
        return hook

    def _router_prehook(self, l: int):
        def hook(module, args):
            h = args[0]           # hidden_states_flat, shape (T, H)
            if isinstance(h, torch.Tensor):
                self.h_post_attn_list[l] = h.detach().float().cpu().numpy()
        return hook

    def _router_outhook(self, l: int):
        def hook(module, args, output):
            # output = (router_probabilities, top_k_weights, top_k_index)
            #   router_probabilities : (T, E)
            #   top_k_weights        : (T, K) normalised
            #   top_k_index          : (T, K) int
            _, weights, indices = output
            if isinstance(indices, torch.Tensor):
                self.true_topk_ids[l]  = indices.detach().cpu().numpy().astype(np.int32)
                self.true_topk_scrs[l] = weights.detach().float().cpu().numpy()
        return hook

    # ─── lifecycle ────────────────────────────────────────────────────────

    def reset(self):
        """Free captured numpy arrays between sequences without removing hooks."""
        self.h_in_list        = [None] * self.n_layers
        self.h_post_attn_list = [None] * self.n_layers
        self.true_topk_ids    = [None] * self.n_layers
        self.true_topk_scrs   = [None] * self.n_layers

    def remove(self):
        """Permanently remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
