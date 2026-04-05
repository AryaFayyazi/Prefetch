"""
Signal comparison experiment for Qwen1.5-MoE-A2.7B.

Compares 5 predictor signals under the same CRC framework:
  A  : baseline — h_post_attn_l  + input_LN_{l+1}          (current system)
  B  : post-MoE — h_in_{l+1}     + post_attn_LN_{l+1}      (correct boundary)
  C  : B + causal EMA routing momentum prior                (routing locality)
  D  : residual velocity extrapolation                      (residual momentum)
  BD : linear blend 0.70·B + 0.30·D                        (combined)

See moe_cp_prefetch/pipeline_qwen_signals.py for the full mathematical description.
"""
import json
import os
from datetime import datetime

from moe_cp_prefetch.config import ExperimentConfig
from moe_cp_prefetch.pipeline_qwen_signals import run_qwen_signals


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

    results = run_qwen_signals(
        cfg,
        n_gamma=400,
        ema_beta=0.30,    # EMA decay for routing momentum prior
        ema_lambda=0.30,  # blend weight for EMA in Signal C
        bd_alpha=0.70,    # weight for Signal B in BD blend
    )

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"qwen_signals_alpha{cfg.alpha}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[run] Results saved → {out_path}")


if __name__ == "__main__":
    main()
