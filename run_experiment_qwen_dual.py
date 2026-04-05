"""
Dual-signal prefetch experiment for Qwen1.5-MoE-A2.7B.

Phase 1 (EARLY):  Signal E_A — velocity-corrected logit blend on post-attention
                  hiddens, ε calibrated by rank-1 hit rate.
                  Prefetches exactly 1 expert per token.

Phase 2 (LATE):   Signal E_B — velocity-corrected logit blend on post-MoE boundary,
                  ε calibrated by rank-4 hit rate.
                  CRC conformal set: covers all K=4 true experts with P ≥ 1−α.

Hybrid:  {top-1(E_A)} ∪ CRC-set(E_B)
  overlap ≈ 99.5% → E_A adds timing advantage at (virtually) zero extra bandwidth.
"""
import json
import os
from datetime import datetime

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars/arrays to native Python types for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from moe_cp_prefetch.config import ExperimentConfig
from moe_cp_prefetch.pipeline_qwen_dual import run_qwen_dual


def main() -> None:
    cfg = ExperimentConfig(
        hf_model_id="Qwen/Qwen1.5-MoE-A2.7B",
        dtype="bfloat16",
        trust_remote_code=False,
        router_top_k=4,
        alpha=0.1,
        dataset_name="allenai/c4",
        dataset_config="en",
        dataset_split_cal="train",
        dataset_split_test="validation",
        text_field="text",
        cache_dir="/datasets",
        max_samples_cal=2000,
        max_samples_test=200,
        max_seq_len=512,
        batch_size=1,
        mondrian_bins=5,
        mondrian_min_bin_size=5000,
        mondrian_binning="rank",
        run_serving_benchmark=False,
        max_new_tokens=64,
        concurrent_requests=4,
        requests_per_worker=10,
        seed=0,
    )

    results = run_qwen_dual(cfg, n_gamma=400)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"qwen_dual_alpha{cfg.alpha}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\n[run] Results saved → {out_path}")


if __name__ == "__main__":
    main()
