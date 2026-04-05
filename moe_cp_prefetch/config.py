from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ExperimentConfig:
    # ---- Model ----
    hf_model_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    dtype: str = "float16"  # Mixtral likes fp16; bf16 also ok on A6000
    trust_remote_code: bool = False
    # ---- No-training predictor ensemble ----
    pred_ensemble: str = "select_min_entropy"   # {"select_min_entropy", "avg_logits"}
    temporal_extrapolate_lambda: float = 1.0    # λ for extrapolation predictor

    # ---- Mondrian CP (robust binning) ----
    mondrian_bins: int = 5
    mondrian_score: str = "entropy"            # {"entropy","margin"}
    mondrian_min_bin_size: int = 5000
    mondrian_binning: str = "rank"             # {"rank","quantile"}  <- USE rank
    # HF cache under /models
    hf_home: str = "/models/hf_cache"
    hf_datasets_cache: str = "/models/hf_cache/datasets"
    hf_hub_cache: str = "/models/hf_cache/hub"

    # ---- MoE routing ----
    router_top_k: int = 2  # Mixtral routes top-2 :contentReference[oaicite:4]{index=4}
    alpha: float = 0.1

    # ---- Trace + simulation ----
    save_traces: bool = True
    trace_out_dir: str = "outputs"
    trace_tag: str = "layerstart"

    # Profiling for simulator calibration
    profile_batches: int = 20          # how many batches to profile for per-layer timings
    profile_seq_len: int = 256         # shorter is faster; should be <= max_seq_len

    # Simulation sweeps
    sim_cache_sizes: tuple[int, ...] = (64, 128, 256, 512, 1024)  # capacity in "experts"
    sim_bandwidth_GBs: tuple[float, ...] = (8.0, 12.0, 16.0, 24.0) # effective H2D GB/s
    sim_prefetch_prioritize_true: bool = False  # keep False for realism (we don't know true)

    # ---- Dataset ----
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    dataset_split_cal: str = "train"
    dataset_split_test: str = "validation"
    text_field: str = "text"
    cache_dir: str = "/datasets"  # datasets cache (HF datasets uses this)

    # These directly control runtime/scale:
    max_samples_cal: int = 2000
    max_samples_test: int = 200
    max_seq_len: int = 512
    batch_size: int = 1

    # ---- Serving benchmark (optional) ----
    run_serving_benchmark: bool = True
    max_new_tokens: int = 64
    concurrent_requests: int = 8
    requests_per_worker: int = 20

    # ---- System metrics ----
    log_every: int = 50
    seed: int = 0

    def asdict(self) -> Dict[str, Any]:
        return dict(self.__dict__)