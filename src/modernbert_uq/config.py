from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    max_length: int
    dropout_prob: float
    pooling_strategy: str
    uq_method: str
    num_ensemble_members: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    num_epochs: int
    cls_loss_weight: float
    difficulty_loss_weight: float
    mc_dropout_passes: int
    output_dir: str
    train_path: str | None = None
    validation_path: str | None = None
    test_path: str | None = None
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    seed: int = 42


def load_config(path: str | Path) -> ExperimentConfig:
    raw_config = _read_yaml(path)
    return ExperimentConfig(
        model_name=str(raw_config["model"]["name"]),
        max_length=int(raw_config["model"]["max_length"]),
        dropout_prob=float(raw_config["model"].get("dropout_prob", 0.1)),
        pooling_strategy=str(raw_config["model"].get("pooling_strategy", "mean")),
        uq_method=str(raw_config["uncertainty"].get("method", "none")),
        num_ensemble_members=int(raw_config["uncertainty"].get("num_ensemble_members", 1)),
        train_batch_size=int(raw_config["training"]["train_batch_size"]),
        eval_batch_size=int(raw_config["training"]["eval_batch_size"]),
        learning_rate=float(raw_config["optimizer"]["learning_rate"]),
        num_epochs=int(raw_config["training"]["num_epochs"]),
        cls_loss_weight=float(raw_config["loss"]["classification_weight"]),
        difficulty_loss_weight=float(raw_config["loss"]["difficulty_weight"]),
        mc_dropout_passes=int(raw_config["uncertainty"]["mc_dropout_passes"]),
        output_dir=str(raw_config["output"]["dir"]),
        train_path=raw_config["data"].get("train_path"),
        validation_path=raw_config["data"].get("validation_path"),
        test_path=raw_config["data"].get("test_path"),
        warmup_ratio=float(raw_config["optimizer"].get("warmup_ratio", 0.0)),
        weight_decay=float(raw_config["optimizer"].get("weight_decay", 0.0)),
        gradient_accumulation_steps=int(
            raw_config["training"].get("gradient_accumulation_steps", 1)
        ),
        mixed_precision=str(raw_config["training"].get("mixed_precision", "no")),
        seed=int(raw_config["training"].get("seed", 42)),
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
