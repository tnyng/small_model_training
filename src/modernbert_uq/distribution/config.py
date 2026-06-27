from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DistributionConfig:
    model_name: str
    max_length: int
    dropout_prob: float
    pooling_strategy: str

    num_score_classes: int
    label_offset: int
    score_target_column: str

    score_loss_weight: float
    difficulty_loss_weight: float
    soft_label_sigma: float

    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    num_epochs: int
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


def load_distribution_config(path: str | Path) -> DistributionConfig:
    raw = _read_yaml(path)
    model_section = raw.get("model", {})
    data_section = raw.get("data", {})
    loss_section = raw.get("loss", {})
    optimizer_section = raw.get("optimizer", {})
    training_section = raw.get("training", {})
    uncertainty_section = raw.get("uncertainty", {})
    output_section = raw.get("output", {})

    num_score_classes = int(model_section.get("num_score_classes", 9))
    if num_score_classes < 2:
        raise ValueError("num_score_classes must be >= 2")

    soft_label_sigma = float(loss_section.get("soft_label_sigma", 0.0))
    if soft_label_sigma < 0:
        raise ValueError("soft_label_sigma must be >= 0")

    return DistributionConfig(
        model_name=str(model_section["name"]),
        max_length=int(model_section["max_length"]),
        dropout_prob=float(model_section.get("dropout_prob", 0.1)),
        pooling_strategy=str(model_section.get("pooling_strategy", "mean")),
        num_score_classes=num_score_classes,
        label_offset=int(model_section.get("label_offset", 1)),
        score_target_column=str(data_section.get("score_target_column", "score_bucket")),
        score_loss_weight=float(loss_section.get("score_weight", 1.0)),
        difficulty_loss_weight=float(loss_section.get("difficulty_weight", 0.0)),
        soft_label_sigma=soft_label_sigma,
        train_batch_size=int(training_section["train_batch_size"]),
        eval_batch_size=int(training_section["eval_batch_size"]),
        learning_rate=float(optimizer_section["learning_rate"]),
        num_epochs=int(training_section["num_epochs"]),
        mc_dropout_passes=int(uncertainty_section.get("mc_dropout_passes", 0)),
        output_dir=str(output_section["dir"]),
        train_path=data_section.get("train_path"),
        validation_path=data_section.get("validation_path"),
        test_path=data_section.get("test_path"),
        warmup_ratio=float(optimizer_section.get("warmup_ratio", 0.0)),
        weight_decay=float(optimizer_section.get("weight_decay", 0.0)),
        gradient_accumulation_steps=int(training_section.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(training_section.get("mixed_precision", "no")),
        seed=int(training_section.get("seed", 42)),
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
