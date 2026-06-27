from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    score_task_type: str
    score_target_column: str
    max_length: int
    dropout_prob: float
    pooling_strategy: str
    uq_method: str
    num_ensemble_members: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    num_epochs: int
    score_loss_weight: float
    difficulty_loss_weight: float
    positive_class_weight: float
    negative_class_weight: float
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
    loss_type: str = "mse"            # "mse" or "bt"
    center_rewards_coeff: float = 0.0  # BT reward-centering regularisation coeff
    pair_score_threshold: float = 0.1  # min |score_a - score_b| to form a pair
    data_fraction: float = 1.0         # fraction of train split to use (0 < f ≤ 1)


def load_config(path: str | Path) -> ExperimentConfig:
    raw_config = _read_yaml(path)
    return ExperimentConfig(
        model_name=str(raw_config["model"]["name"]),
        score_task_type=str(raw_config["model"].get("score_task_type", "classification")),
        score_target_column=str(raw_config["data"].get("score_target_column", "label")),
        max_length=int(raw_config["model"]["max_length"]),
        dropout_prob=float(raw_config["model"].get("dropout_prob", 0.1)),
        pooling_strategy=str(raw_config["model"].get("pooling_strategy", "mean")),
        uq_method=str(raw_config["uncertainty"].get("method", "none")),
        num_ensemble_members=int(raw_config["uncertainty"].get("num_ensemble_members", 1)),
        train_batch_size=int(raw_config["training"]["train_batch_size"]),
        eval_batch_size=int(raw_config["training"]["eval_batch_size"]),
        learning_rate=float(raw_config["optimizer"]["learning_rate"]),
        num_epochs=int(raw_config["training"]["num_epochs"]),
        score_loss_weight=float(
            raw_config["loss"].get(
                "score_weight",
                raw_config["loss"].get("classification_weight", 1.0),
            )
        ),
        difficulty_loss_weight=float(raw_config["loss"]["difficulty_weight"]),
        positive_class_weight=float(raw_config["loss"].get("positive_class_weight", 1.0)),
        negative_class_weight=float(raw_config["loss"].get("negative_class_weight", 1.0)),
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
        loss_type=str(raw_config["loss"].get("type", "mse")),
        center_rewards_coeff=float(raw_config["loss"].get("center_rewards_coeff", 0.0)),
        pair_score_threshold=float(raw_config["data"].get("pair_score_threshold", 0.1)),
        data_fraction=float(raw_config["data"].get("fraction", 1.0)),
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
