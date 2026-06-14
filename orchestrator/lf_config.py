"""Generate LLaMA-Factory inputs (dataset_info.json, training YAML, optional
DeepSpeed config) from the orchestrator config + per-run overrides."""
from __future__ import annotations

import json
from typing import Any

import yaml

# Maps our stored OpenAI-style `messages` JSONL to a LLaMA-Factory dataset.
# Each training line looks like: {"messages":[{"role":"user","content":...}, ...]}
DATASET_NAME = "ten31"


def dataset_info(file_name: str) -> dict[str, Any]:
    return {
        DATASET_NAME: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def train_yaml(cfg: dict, *, multinode: bool = False) -> str:
    """Build a LLaMA-Factory train config. Paths are RemotePaths inside the
    container on the Spark, where the run dir is mounted at /workspace/run."""
    out: dict[str, Any] = {
        "model_name_or_path": cfg.get("model", "google/gemma-4-31B-it"),
        "trust_remote_code": True,
        # --- method ---
        "stage": "sft",
        "do_train": True,
        "finetuning_type": cfg.get("finetuningType", "lora"),
        # --- data ---
        "dataset": DATASET_NAME,
        "dataset_dir": "/workspace/run",
        "template": cfg.get("template", "gemma4"),
        "cutoff_len": int(cfg.get("cutoffLen", 4096)),
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        # --- output ---
        "output_dir": "/workspace/run/output",
        "logging_steps": 1,
        "save_steps": 100,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        # --- train ---
        "per_device_train_batch_size": int(cfg.get("perDeviceBatchSize", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradAccumSteps", 8)),
        "learning_rate": float(cfg.get("learningRate", "1.0e-4")),
        "num_train_epochs": float(cfg.get("epochs", 3)),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
    }

    if out["finetuning_type"] == "lora":
        out["lora_target"] = "all"
        out["lora_rank"] = int(cfg.get("loraRank", 16))
        out["lora_alpha"] = int(cfg.get("loraAlpha", 32))

    qb = cfg.get("quantizationBit")
    if qb in (4, 8):
        out["quantization_bit"] = int(qb)
        out["quantization_method"] = "bitsandbytes"

    if multinode:
        # ZeRO-3 across the two Sparks; entrypoint sets FORCE_TORCHRUN/NNODES.
        out["deepspeed"] = "/workspace/run/ds_z3.json"

    return yaml.safe_dump(out, sort_keys=False)


def deepspeed_z3() -> str:
    cfg = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }
    return json.dumps(cfg, indent=2)
