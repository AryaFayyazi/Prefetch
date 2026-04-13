"""
Dual-signal prefetch experiment for google/gemma-4-26B-A4B.

Phase 1 (EARLY):  Signal E_A  — velocity-corrected router-logit prediction from
                  h_in_l (output of layer l-1), ε calibrated by rank-1 hit rate.
                  Prefetches 1 expert ahead of the FFN, during the attention phase.

Phase 2 (EXACT):  Signal E_B  — velocity-corrected router-logit prediction from
                  h_post_attn_l (exact router input), ε calibrated by rank-8 hit rate.
                  CRC conformal set: covers all K=8 true experts with E[loss] ≤ α.

Hybrid: {top-1(E_A)} ∪ CRC-set(E_B)
  overlap measures how often E_A's early pick is already in E_B's conformal set.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

# ── Offline mode + dataset cache ──────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/models/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/datasets")

# Make sure the project root is importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe_cp_prefetch.pipeline_gemma4_dual import run_gemma4_dual


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars / arrays to native Python for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main() -> None:
    results = run_gemma4_dual(
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

    # ── Per-layer summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[run] Per-layer calibration summary (Signal E_A | E_B)")
    print(f"  {'lyr':>3}  {'ε_A':>6}  {'ε_B':>6}  {'γ*':>8}")
    print("  " + "-" * 35)
    for l, (eps_a, eps_b, gstar) in enumerate(zip(
        results["best_eps_EA"],
        results["best_eps_EB"],
        results["gamma_stars"],
    )):
        print(f"  {l:>3d}  {eps_a:>6.2f}  {eps_b:>6.2f}  {gstar:>8.5f}")

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
        print(f"  avg |S|   = {metrics_eb.get('avg_set', float('nan')):.2f}")
        print(f"  E[loss]   = {metrics_eb.get('mean_L',  float('nan')):.4f}")
        for k in [1, 8]:
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
    out_path = os.path.join(out_dir, f"gemma4_dual_alpha{results.get('alpha', 0.10)}_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)

    print(f"\n[run] Results saved → {out_path}")


if __name__ == "__main__":
    main()
