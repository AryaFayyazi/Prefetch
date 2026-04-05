from __future__ import annotations

"""
Adapter for Qwen2MoeForCausalLM  (Qwen1.5-MoE, etc.) — Post-Attention signal
───────────────────────────────────────────────────────────────────────────────
Prediction strategy ("post-attention" / pre-MLP hop-1)
  • Signal used  : h_post_attn_ℓ  — hidden state right AFTER attention + residual,
                   i.e. the input to post_attention_layernorm at layer ℓ.
                   This is captured via a pre-hook on layer.post_attention_layernorm.
  • Target       : top-4 routed experts of layer ℓ+1  (NEXT layer)
  • Approx logits: W_gate_{ℓ+1}  @  h_post_attn_ℓ.T   (no layernorm applied)

Motivation
  Using h_post_attn rather than h_in (layerstart) gives a slightly more
  processed signal (self-attention has already run), potentially yielding
  better prediction.  The cross-layer aspect (predicting layer l+1 from
  layer l) means we have the full attention window to prefetch experts.

Architecture details
  Qwen2MoeDecoderLayer forward (simplified):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, ... = self.self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states          ← h_post_attn  (captured here)
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)           ← true_topk captured here
    hidden_states = residual + hidden_states          ← h_out = h_in of next layer
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PostAttnTrace:
    post_attn_states: torch.Tensor  # CPU, (T, H) — h after attn+residual (before post_attn_norm+MLP)
    true_topk_ids: torch.Tensor     # CPU, (T, K) — true router top-k indices AT THIS LAYER
    num_experts: int                # E  (60 for Qwen1.5-MoE; 0 for dense layers)
    top_k: int                      # K  (4 for Qwen1.5-MoE)
    hidden_dim: int                 # H
    true_topk_scores: Optional[torch.Tensor] = None  # CPU, (T, K) — re-normalised gate weights
                                                      # i.e. softmax(logits)[top-K] / sum,
                                                      # exactly the weights used in MoE compute.
    layer_input_states: Optional[torch.Tensor] = None  # CPU, (T, H) — h_in_l = residual stream
                                                        # BEFORE input_layernorm of this layer;
                                                        # = output of the PREVIOUS decoder layer.
                                                        # Used as Signal B source for predicting
                                                        # THIS layer's experts from the layer before.


class QwenMoePostAttnTracer:
    """
    Attaches lightweight hooks to every Qwen2MoeDecoderLayer:
      1. Pre-hook  on layer.post_attention_layernorm → captures h_post_attn_ℓ
      2. Post-hook on layer.mlp                      → captures router_logits
                                                       → derives true top-k at ℓ

    The pipeline pairs: signal  = post_attn_states[l]
                        gate    = W_gate[l+1]
                        target  = true_topk_ids[l+1]
    for l = 0 .. num_layers-2.

    Call `pop_last_forward(seq_len)` after each forward pass.
    Call `detach()` before deleting or re-fitting.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        self._hooks: List[Any] = []
        self._layer_modules: List[nn.Module] = []
        self._moe_blocks: List[Optional[nn.Module]] = []
        self._records: Dict[int, Dict[str, torch.Tensor]] = {}
        self._router_W_eh: Dict[int, torch.Tensor] = {}   # cache

    # ------------------------------------------------------------------ attach
    def attach(self, model: nn.Module) -> None:
        self.detach()

        decoder_layers: List[nn.Module] = []
        for _, m in model.named_modules():
            if m.__class__.__name__ == "Qwen2MoeDecoderLayer":
                decoder_layers.append(m)
        if not decoder_layers:
            raise RuntimeError(
                "Could not find Qwen2MoeDecoderLayer modules. "
                "Is this a Qwen2MoeForCausalLM from HF transformers?"
            )

        self._layer_modules = decoder_layers
        self._records = {}
        self._router_W_eh = {}
        self._moe_blocks = []

        for li, layer in enumerate(self._layer_modules):
            self._records[li] = {}

            # ── 0. Pre-hook on input_layernorm: captures h_in_l (layer boundary signal) ─
            input_ln = getattr(layer, "input_layernorm", None)
            if input_ln is None:
                raise RuntimeError(
                    f"Layer {li} has no .input_layernorm attribute. "
                    "Is this a non-standard Qwen2Moe variant?"
                )

            def _make_input_pre_hook(layer_idx: int):
                def pre_hook(module: nn.Module, inputs: Tuple[Any, ...]):
                    hs = inputs[0]
                    if not torch.is_tensor(hs) or hs.ndim != 3:
                        return
                    # batch=1; take [0] to get (T, H)
                    self._records[layer_idx]["layer_input_states"] = hs.detach().cpu()[0]
                return pre_hook

            self._hooks.append(
                input_ln.register_forward_pre_hook(_make_input_pre_hook(li))
            )

            # ── 1. Pre-hook on post_attention_layernorm: captures h_post_attn ─
            post_ln = getattr(layer, "post_attention_layernorm", None)
            if post_ln is None:
                raise RuntimeError(
                    f"Layer {li} has no .post_attention_layernorm attribute. "
                    "Is this a non-standard Qwen2Moe variant?"
                )

            def _make_post_attn_pre_hook(layer_idx: int):
                def pre_hook(module: nn.Module, inputs: Tuple[Any, ...]):
                    hs = inputs[0]
                    if not torch.is_tensor(hs) or hs.ndim != 3:
                        # Non-3-D call — this module may be invoked directly
                        # (e.g. from _gate_probs with an explicit 3-D wrap);
                        # skip the capture silently to avoid interfering.
                        return
                    # batch=1; take [0] to get (T, H)
                    self._records[layer_idx]["post_attn_states"] = hs.detach().cpu()[0]
                return pre_hook

            self._hooks.append(
                post_ln.register_forward_pre_hook(_make_post_attn_pre_hook(li))
            )

            # ── 2. Find mlp (Qwen2MoeSparseMoeBlock or dense) ─────────────────
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                raise RuntimeError(f"Layer {li} has no .mlp attribute.")
            self._moe_blocks.append(mlp)

            if mlp.__class__.__name__ != "Qwen2MoeSparseMoeBlock":
                # Dense layer — no routing
                continue

            # ── 3. Post-hook on mlp: captures router_logits → true top-k ──────
            def _make_mlp_post_hook(layer_idx: int):
                def post_hook(
                    module: nn.Module,
                    inputs: Tuple[Any, ...],
                    output: Any,
                ):
                    if not (isinstance(output, (tuple, list)) and len(output) == 2):
                        raise RuntimeError(
                            f"[QwenPostAttnTracer] layer {layer_idx}: expected "
                            f"(hidden_states, router_logits), got {type(output)}"
                        )
                    router_logits = output[1]  # (T, E)  float32
                    routing_weights_full = F.softmax(router_logits.float(), dim=-1)  # (T, E)
                    topk_scores, selected_experts = torch.topk(
                        routing_weights_full, self.top_k, dim=-1
                    )  # both (T, K)
                    # Re-normalise exactly as Qwen's SparseMoeBlock does:
                    #   routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)
                    topk_scores_norm = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # (T, K)
                    self._records[layer_idx]["true_topk_ids"] = selected_experts.detach().cpu()
                    self._records[layer_idx]["true_topk_scores"] = topk_scores_norm.detach().cpu()
                return post_hook

            self._hooks.append(mlp.register_forward_hook(_make_mlp_post_hook(li)))

    # ----------------------------------------------------------------- detach
    def detach(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self._layer_modules = []
        self._moe_blocks = []
        self._records = {}
        self._router_W_eh = {}

    # ----------------------------------------------------------- num_layers
    def num_layers(self) -> int:
        return len(self._layer_modules)

    # --------------------------------------------------- get_router_weight_EH
    def get_router_weight_EH(self, layer_idx: int) -> torch.Tensor:
        """
        Returns gate weight W of shape (E, H).
        Logits = h @ W.T  →  argmax gives predicted expert.
        """
        if layer_idx in self._router_W_eh:
            return self._router_W_eh[layer_idx]

        mlp = self._moe_blocks[layer_idx]
        if mlp is None or mlp.__class__.__name__ != "Qwen2MoeSparseMoeBlock":
            raise RuntimeError(f"Layer {layer_idx} is not a sparse MoE block.")

        gate: nn.Linear = mlp.gate  # nn.Linear(H, E, bias=False)
        W = gate.weight             # (E, H)
        if W.shape[0] > W.shape[1]:
            W = W.t()
        self._router_W_eh[layer_idx] = W
        return W

    # ------------------------------------------------- get_input_layernorm
    def get_input_layernorm(self, layer_idx: int) -> nn.Module:
        """
        Returns the input_layernorm (RMSNorm) of decoder layer `layer_idx`.
        This is the LN applied to h_in_l before the self-attention block.
        Used in Signal A (baseline) as an approximate LN for gate projection.
        """
        if layer_idx >= len(self._layer_modules) or not self._layer_modules:
            raise RuntimeError(
                f"[QwenPostAttnTracer] Layer {layer_idx} not found — "
                "did you call attach() first?"
            )
        ln = getattr(self._layer_modules[layer_idx], "input_layernorm", None)
        if ln is None:
            raise RuntimeError(
                f"[QwenPostAttnTracer] Layer {layer_idx} has no input_layernorm."
            )
        return ln

    # ---------------------------------------------- get_post_attn_layernorm
    def get_post_attn_layernorm(self, layer_idx: int) -> nn.Module:
        """
        Returns the post_attention_layernorm (RMSNorm) of decoder layer `layer_idx`.
        This is the LN applied after attention+residual, immediately before the MoE
        gate — i.e. the ACTUAL normalisation the gate operates on:
            gate_logits = W_gate @ post_attn_LN(h_post_attn)
        Used in Signals B, C, D as the correct LN for gate projection.
        """
        if layer_idx >= len(self._layer_modules) or not self._layer_modules:
            raise RuntimeError(
                f"[QwenPostAttnTracer] Layer {layer_idx} not found — "
                "did you call attach() first?"
            )
        ln = getattr(self._layer_modules[layer_idx], "post_attention_layernorm", None)
        if ln is None:
            raise RuntimeError(
                f"[QwenPostAttnTracer] Layer {layer_idx} has no post_attention_layernorm."
            )
        return ln

    # -------------------------------------------------- pop_last_forward
    def pop_last_forward(self, seq_len: int) -> List[PostAttnTrace]:
        """
        Collect traces from the most-recent forward pass and return one
        PostAttnTrace per layer (in layer order).  Clears internal records.
        """
        traces: List[PostAttnTrace] = []

        for li in range(self.num_layers()):
            rec = self._records.get(li, {})

            if "post_attn_states" not in rec:
                raise RuntimeError(
                    f"[QwenPostAttnTracer] Missing post_attn_states for layer {li}. "
                    "Did you call model.forward() with hooks attached?"
                )

            post_attn = rec["post_attn_states"]  # (T, H)
            if post_attn.shape[0] != seq_len:
                raise RuntimeError(
                    f"[QwenPostAttnTracer] post_attn_states seq_len mismatch "
                    f"layer {li}: got {post_attn.shape[0]}, expected {seq_len}"
                )

            layer_input_dense = rec.get("layer_input_states", None)

            mlp = self._moe_blocks[li]
            if mlp is None or mlp.__class__.__name__ != "Qwen2MoeSparseMoeBlock":
                traces.append(PostAttnTrace(
                    post_attn_states=post_attn,
                    true_topk_ids=torch.empty((seq_len, 0), dtype=torch.long),
                    num_experts=0,
                    top_k=0,
                    hidden_dim=post_attn.shape[1],
                    layer_input_states=layer_input_dense,
                ))
                continue

            if "true_topk_ids" not in rec:
                raise RuntimeError(
                    f"[QwenPostAttnTracer] Missing true_topk_ids for MoE layer {li}."
                )

            topk = rec["true_topk_ids"]  # (T, K)
            if topk.shape[0] != seq_len:
                raise RuntimeError(
                    f"[QwenPostAttnTracer] true_topk_ids seq_len mismatch "
                    f"layer {li}: got {topk.shape[0]}, expected {seq_len}"
                )

            W_eh = self.get_router_weight_EH(li)
            E = int(W_eh.shape[0])
            H = int(W_eh.shape[1])

            topk_scores = rec.get("true_topk_scores", None)  # (T, K) or None
            if topk_scores is not None and topk_scores.shape[0] != seq_len:
                raise RuntimeError(
                    f"[QwenPostAttnTracer] true_topk_scores seq_len mismatch "
                    f"layer {li}: got {topk_scores.shape[0]}, expected {seq_len}"
                )

            layer_input = rec.get("layer_input_states", None)  # (T, H) or None
            if layer_input is not None and layer_input.shape[0] != seq_len:
                raise RuntimeError(
                    f"[QwenPostAttnTracer] layer_input_states seq_len mismatch "
                    f"layer {li}: got {layer_input.shape[0]}, expected {seq_len}"
                )

            traces.append(PostAttnTrace(
                post_attn_states=post_attn,
                true_topk_ids=topk,
                num_experts=E,
                top_k=self.top_k,
                hidden_dim=H,
                true_topk_scores=topk_scores,
                layer_input_states=layer_input,
            ))

        # Clear records for next forward
        for li in self._records:
            self._records[li] = {}

        return traces
