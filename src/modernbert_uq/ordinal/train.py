from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modernbert_uq.data import (
    TextDifficultyDataset,
    load_records,
    tokenize_records,
)
from modernbert_uq.utils import ensure_directory, move_batch_to_device

from modernbert_uq.distribution.data import DistributionBatchCollator

from modernbert_uq.ordinal.config import OrdinalConfig, load_ordinal_config
from modernbert_uq.ordinal.evaluate import evaluate_ordinal_predictions
from modernbert_uq.ordinal.modeling import ModernBertOrdinalModel
from modernbert_uq.ordinal.predict import export_ordinal_prediction_table
from modernbert_uq.ordinal.uncertainty import mc_dropout_ordinal_predict


@dataclass(frozen=True)
class OrdinalTrainingResult:
    global_step: int
    best_eval_loss: float | None
    artifact_path: Path


def run_ordinal_experiment(config: OrdinalConfig) -> OrdinalTrainingResult:
    from transformers import AutoTokenizer

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = ensure_directory(config.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    collator = DistributionBatchCollator(
        num_score_classes=config.num_score_classes,
        label_offset=config.label_offset,
    )

    train_loader, _train_records = _build_loader(
        config.train_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.train_batch_size,
        shuffle=True,
        target_column=config.score_target_column,
        collator=collator,
    )
    validation_loader, validation_records = _build_loader(
        config.validation_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.eval_batch_size,
        shuffle=False,
        target_column=config.score_target_column,
        collator=collator,
    )
    test_loader, test_records = _build_loader(
        config.test_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.eval_batch_size,
        shuffle=False,
        target_column=config.score_target_column,
        collator=collator,
    )
    if train_loader is None:
        raise ValueError("Training data is required")

    model = ModernBertOrdinalModel(
        model_name=config.model_name,
        num_score_classes=config.num_score_classes,
        label_offset=config.label_offset,
        dropout_prob=config.dropout_prob,
        pooling_strategy=config.pooling_strategy,
        score_loss_weight=config.score_loss_weight,
        difficulty_loss_weight=config.difficulty_loss_weight,
    )

    result = _run_training(
        model=model,
        train_batches=list(train_loader),
        eval_batches=list(validation_loader) if validation_loader is not None else None,
        config=config,
        output_dir=output_dir,
    )

    if validation_loader is not None:
        _run_split(
            model=model,
            loader=validation_loader,
            records=validation_records,
            split_name="validation",
            output_dir=output_dir,
            config=config,
        )
    if test_loader is not None:
        _run_split(
            model=model,
            loader=test_loader,
            records=test_records,
            split_name="test",
            output_dir=output_dir,
            config=config,
        )

    return result


def _build_loader(
    path: str | Path | None,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    target_column: str,
    collator: DistributionBatchCollator,
) -> tuple[DataLoader | None, list[dict[str, object]]]:
    if not path:
        return None, []
    records = load_records(path, target_column=target_column)
    tokenized = tokenize_records(
        records,
        tokenizer=tokenizer,
        max_length=max_length,
        target_column=target_column,
    )
    dataset = TextDifficultyDataset(tokenized)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
    return dataloader, records


def _run_training(
    model: torch.nn.Module,
    train_batches: Iterable[dict[str, torch.Tensor]],
    eval_batches: Iterable[dict[str, torch.Tensor]] | None,
    config: OrdinalConfig,
    output_dir: Path,
) -> OrdinalTrainingResult:
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    train_batches = list(train_batches)
    eval_batches = list(eval_batches or [])

    global_step = 0
    best_eval_loss: float | None = None

    for _ in range(config.num_epochs):
        model.train()
        for batch in train_batches:
            batch = move_batch_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                outputs = model(**batch)
                if outputs.loss is None:
                    raise ValueError("Training batches must produce a loss")
                accelerator.backward(outputs.loss)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1

        if eval_batches:
            eval_loss = _evaluate_loss(model, eval_batches, accelerator.device)
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss

    artifact_path = output_dir / "training_state.pt"
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(
        {
            "model_state_dict": unwrapped.state_dict(),
            "global_step": global_step,
            "best_eval_loss": best_eval_loss,
        },
        artifact_path,
    )
    return OrdinalTrainingResult(
        global_step=global_step,
        best_eval_loss=best_eval_loss,
        artifact_path=artifact_path,
    )


def _evaluate_loss(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved_batch = move_batch_to_device(batch, device)
            outputs = model(**moved_batch)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def predict_ordinal_batches(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()

    threshold_logits_chunks: list[np.ndarray] = []
    cumulative_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    expected_chunks: list[np.ndarray] = []
    argmax_chunks: list[np.ndarray] = []
    threshold_chunks: list[np.ndarray] = []
    entropy_chunks: list[np.ndarray] = []
    difficulty_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    has_difficulty_head = False

    with torch.no_grad():
        for batch in batches:
            moved_batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=moved_batch["input_ids"],
                attention_mask=moved_batch.get("attention_mask"),
            )
            threshold_logits_chunks.append(outputs.threshold_logits.detach().cpu().numpy())
            cumulative_chunks.append(outputs.cumulative_gt.detach().cpu().numpy())
            probability_chunks.append(outputs.class_probabilities.detach().cpu().numpy())
            expected_chunks.append(outputs.expected_score.detach().cpu().numpy())
            argmax_chunks.append(outputs.argmax_score.detach().cpu().numpy())
            threshold_chunks.append(outputs.threshold_score.detach().cpu().numpy())
            entropy_chunks.append(outputs.predictive_entropy.detach().cpu().numpy())
            if outputs.difficulty_pred is not None:
                has_difficulty_head = True
                difficulty_chunks.append(outputs.difficulty_pred.detach().cpu().numpy())
            if "labels" in batch:
                label_chunks.append(batch["labels"].detach().cpu().numpy())

    result = {
        "threshold_logits": np.concatenate(threshold_logits_chunks, axis=0),
        "cumulative_gt": np.concatenate(cumulative_chunks, axis=0),
        "class_probabilities": np.concatenate(probability_chunks, axis=0),
        "expected_score": np.concatenate(expected_chunks, axis=0),
        "argmax_score": np.concatenate(argmax_chunks, axis=0),
        "threshold_score": np.concatenate(threshold_chunks, axis=0),
        "predictive_entropy": np.concatenate(entropy_chunks, axis=0),
    }
    if has_difficulty_head:
        result["difficulty_predictions"] = np.concatenate(difficulty_chunks, axis=0)
    if label_chunks:
        result["class_indices"] = np.concatenate(label_chunks, axis=0)
    return result


def _run_split(
    model: torch.nn.Module,
    loader: DataLoader,
    records: list[dict[str, object]],
    split_name: str,
    output_dir: Path,
    config: OrdinalConfig,
) -> None:
    if not records:
        return

    predictions = predict_ordinal_batches(model, list(loader))

    ids = [str(record.get("id", f"{split_name}_{index}")) for index, record in enumerate(records)]
    gold_scores = np.asarray(
        [record[config.score_target_column] for record in records],
        dtype=float,
    )
    gold_difficulty = np.asarray([record["difficulty"] for record in records], dtype=float)
    accuracy_scores = None
    if "accuracy_score" in records[0]:
        accuracy_scores = np.asarray(
            [record["accuracy_score"] for record in records],
            dtype=float,
        )

    difficulty_predictions = predictions.get("difficulty_predictions")
    table = export_ordinal_prediction_table(
        ids=ids,
        class_probabilities=predictions["class_probabilities"],
        cumulative_gt=predictions["cumulative_gt"],
        expected_scores=predictions["expected_score"],
        argmax_scores=predictions["argmax_score"],
        threshold_scores=predictions["threshold_score"],
        predictive_entropy=predictions["predictive_entropy"],
        gold_scores=gold_scores,
        predicted_difficulty=difficulty_predictions,
        gold_difficulty=gold_difficulty,
        label_offset=config.label_offset,
        accuracy_scores=accuracy_scores,
    )
    table.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    metrics = evaluate_ordinal_predictions(
        gold_scores=gold_scores,
        class_probabilities=predictions["class_probabilities"],
        cumulative_gt=predictions["cumulative_gt"],
        expected_scores=predictions["expected_score"],
        argmax_scores=predictions["argmax_score"],
        threshold_scores=predictions["threshold_score"],
        label_offset=config.label_offset,
        difficulty_predictions=difficulty_predictions,
        difficulty_targets=gold_difficulty if difficulty_predictions is not None else None,
    )
    (output_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    if config.mc_dropout_passes > 0:
        _save_mc_dropout(
            model=model,
            loader=loader,
            records=records,
            split_name=split_name,
            output_dir=output_dir,
            config=config,
        )


def _save_mc_dropout(
    model: torch.nn.Module,
    loader: DataLoader,
    records: list[dict[str, object]],
    split_name: str,
    output_dir: Path,
    config: OrdinalConfig,
) -> None:
    import pandas as pd

    device = next(model.parameters()).device
    bin_values = torch.arange(
        config.label_offset,
        config.label_offset + config.num_score_classes,
        dtype=torch.float32,
    )

    rows: list[dict[str, object]] = []
    offset = 0
    for batch in loader:
        moved_batch = move_batch_to_device(batch, device)
        mc_outputs = mc_dropout_ordinal_predict(
            model=model,
            batch=moved_batch,
            num_passes=config.mc_dropout_passes,
            bin_values=bin_values,
        )
        batch_size = moved_batch["input_ids"].shape[0]
        mean_probability = mc_outputs["mean_probability"].detach().cpu().numpy()
        probability_std = mc_outputs["probability_std"].detach().cpu().numpy()
        mean_cumulative = mc_outputs["mean_cumulative_gt"].detach().cpu().numpy()
        cumulative_std = mc_outputs["cumulative_gt_std"].detach().cpu().numpy()
        for index in range(batch_size):
            record = records[offset + index]
            row: dict[str, object] = {
                "id": record.get("id", f"{split_name}_{offset + index}"),
                config.score_target_column: record[config.score_target_column],
                "gold_difficulty": record["difficulty"],
                "mean_expected_score": float(mc_outputs["mean_expected_score"][index].detach().cpu()),
                "expected_score_variance": float(
                    mc_outputs["expected_score_variance"][index].detach().cpu()
                ),
                "predictive_entropy": float(mc_outputs["predictive_entropy"][index].detach().cpu()),
                "expected_entropy": float(mc_outputs["expected_entropy"][index].detach().cpu()),
                "mutual_information": float(mc_outputs["mutual_information"][index].detach().cpu()),
                "argmax_score_from_mean": float(
                    mc_outputs["argmax_score_from_mean"][index].detach().cpu()
                ),
            }
            for class_index in range(config.num_score_classes):
                column = class_index + config.label_offset
                row[f"mean_p_{column}"] = float(mean_probability[index, class_index])
                row[f"std_p_{column}"] = float(probability_std[index, class_index])
            for threshold_index in range(config.num_score_classes - 1):
                threshold = threshold_index + config.label_offset
                row[f"mean_p_gt_{threshold}"] = float(mean_cumulative[index, threshold_index])
                row[f"std_p_gt_{threshold}"] = float(cumulative_std[index, threshold_index])
            if "mean_difficulty" in mc_outputs:
                row["mean_difficulty"] = float(mc_outputs["mean_difficulty"][index].detach().cpu())
                row["difficulty_variance"] = float(
                    mc_outputs["difficulty_variance"][index].detach().cpu()
                )
            rows.append(row)
        offset += batch_size

    artifact_dir = ensure_directory(output_dir / "mc_dropout")
    pd.DataFrame(rows).to_csv(
        artifact_dir / f"{split_name}_mc_dropout_predictions.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ModernBERT CORN ordinal head from a YAML config."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_ordinal_config(args.config)
    run_ordinal_experiment(config)


if __name__ == "__main__":
    main()
