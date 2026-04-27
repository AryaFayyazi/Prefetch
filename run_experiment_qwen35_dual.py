"""
Dual-signal prefetch experiment for Qwen/Qwen3.5-35B-A3B.

Phase 1 (EARLY):  Signal E_A — velocity-corrected router-logit prediction from
                  h_in_l (output of layer l-1), ε calibrated by rank-1 hit rate.
                  Fires during attention_l, before MoE gate runs.

Phase 2 (EXACT):  Signal E_B — velocity-corrected router-logit prediction from
                  h_pre_router_l (exact post-LN gate input), ε calibrated by
                  rank-K hit rate.  CRC conformal set: E[loss] ≤ α guaranteed.

Hybrid: {top-1(E_A)} ∪ CRC-set(E_B)
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME",            "/models/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE",  "/datasets")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe_cp_prefetch.pipeline_qwen35_dual import run_qwen35_dual


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def main() -> None:
    results = run_qwen35_dual(
        alpha=0.10,
        dataset_name="allenai/c4",
        dataset_config="en",
        split_cal="train",
        split_test="validation",
        cache_dir="/datasets",
        max_samples_cal=2000,
        max_samples_test=200,
        max_seq_len=512,
        text_field="text",
        seed=0,
        n_gamma=400,
    )

    # ── Per-layer calibration summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[run] Per-layer calibration summary (Signal E_A | E_B)")
    K        = results["K"]
    n_layers = results["n_layers"]
    print(f"  {'lyr':>5}  {'ε_A':>6}  {'ε_B':>6}  {'γ*':>8}")
    print("  " + "-" * 35)
    for l, (eps_a, eps_b, gstar) in enumerate(zip(
        results["best_eps_EA"], results["best_eps_EB"], results["gamma_stars"],
    )):
        src  = (l - 1) if l > 0 else (n_layers - 1)
        flag = "*" if l == 0 else " "
        print(f"  {src:02d}→{l:02d}{flag}  {eps_a:>6.2f}  {eps_b:>6.2f}  {gstar:>8.5f}")

    # ── Aggregated test metrics ───────────────────────────────────────────────
    metrics_ea = results.get("metrics_EA", {})
    metrics_eb = results.get("metrics_EB", {})
    metrics_d  = results.get("metrics_dual", {})

    print("\n[run] Signal E_A — rank hit rates:")
    rh_ea = metrics_ea.get("rank_hit", {})
    for k in [1, 2, 4, 8]:
        v = rh_ea.get(k)
        if v is not None:
            print(f"  top-{k}: {v:.4f}")

    print("\n[run] Signal E_B — CRC set metrics:")
    if metrics_eb:
        rh_eb = metrics_eb.get("rank_hit", {})
        print(f"  avg |S|   = {metrics_eb.get('avg_set',  float('nan')):.2f}")
        print(f"  E[loss]   = {metrics_eb.get('mean_L',   float('nan')):.4f}")
        for k in [1, K]:
            v = rh_eb.get(k)
            if v is not None:
                print(f"  rank-{k} hit = {v:.4f}")

    print("\n[run] Hybrid (E_A ∪ E_B) metrics:")
    if metrics_d:
        print(f"  overlap   = {metrics_d.get('overlap_rate', float('nan')):.4f}")
        print(f"  avg|union|= {metrics_d.get('avg_union',    float('nan')):.2f}")
        print(f"  E_B cov   = {metrics_d.get('eb_coverage',  float('nan')):.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"qwen35_dual_alpha{results.get('alpha', 0.10)}_{ts}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\n[run] Results saved → {out_path}")


if __name__ == "__main__":
    main()
