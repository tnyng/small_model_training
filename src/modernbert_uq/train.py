from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modernbert_uq.config import ExperimentConfig, load_config
from modernbert_uq.data import (
    TextDifficultyDataset,
    UQBatchCollator,
    PairDataset,
    PairBatchCollator,
    PairCsvIterableDataset,
    PairTextBatchCollator,
    load_pair_records,
    tokenize_pair_records,
    load_records,
    tokenize_records,
)
from modernbert_uq.evaluate import evaluate_predictions
from modernbert_uq.modeling import ModernBertUQModel
from modernbert_uq.predict import export_prediction_table
from modernbert_uq.uncertainty import mc_dropout_predict

from modernbert_uq.utils import ensure_directory, move_batch_to_device


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    num_epochs: int
    output_dir: str | Path
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingResult:
    global_step: int
    best_eval_loss: float | None
    artifact_path: Path


def run_training(
    model: torch.nn.Module,
    train_batches: Iterable[dict[str, torch.Tensor]],
    eval_batches: Iterable[dict[str, torch.Tensor]] | None,
    config: TrainConfig,
) -> TrainingResult:
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    output_dir = ensure_directory(config.output_dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0
    best_eval_loss = None
    train_batches = list(train_batches)
    eval_batches = list(eval_batches or [])

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
            eval_loss = evaluate_loss(model, eval_batches, accelerator.device)
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
    return TrainingResult(
        global_step=global_step,
        best_eval_loss=best_eval_loss,
        artifact_path=artifact_path,
    )


def run_bt_training(
    model: torch.nn.Module,
    train_pair_batches: Iterable[dict[str, torch.Tensor]],
    eval_batches: Iterable[dict[str, torch.Tensor]] | None,
    config: TrainConfig,
    use_margin: bool = False,
    center_rewards_coeff: float = 0.0,
) -> TrainingResult:
    """Training loop for Bradley-Terry objective using pair batches."""
    from modernbert_uq.losses import bradley_terry_loss

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    output_dir = ensure_directory(config.output_dir)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0
    best_eval_loss = None

    for epoch_idx in range(config.num_epochs):
        model.train()
        batch_count = 0
        for batch in train_pair_batches:
            batch_count += 1
            batch = move_batch_to_device(batch, accelerator.device)
            margin = batch.get("margin") if use_margin else None
            with accelerator.accumulate(model):
                out_a = model(
                    input_ids=batch["input_ids_a"],
                    attention_mask=batch["attention_mask_a"],
                )
                out_b = model(
                    input_ids=batch["input_ids_b"],
                    attention_mask=batch["attention_mask_b"],
                )
                uq_method = getattr(accelerator.unwrap_model(model), "uq_method", "none")
                if uq_method == "head_ensemble" and out_a.member_scores is not None:
                    n_members = out_a.member_scores.shape[1]
                    bt_losses = [
                        bradley_terry_loss(
                            out_a.member_scores[:, k],
                            out_b.member_scores[:, k],
                            margin=margin,
                            center_rewards_coeff=center_rewards_coeff,
                        )
                        for k in range(n_members)
                    ]
                    loss = torch.stack(bt_losses).mean()
                else:
                    loss = bradley_terry_loss(
                        out_a.score_pred,
                        out_b.score_pred,
                        margin=margin,
                        center_rewards_coeff=center_rewards_coeff,
                    )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if batch_count % 1000 == 0:
                accelerator.print(
                    f"[BT] epoch={epoch_idx + 1}/{config.num_epochs} "
                    f"batch={batch_count} global_step={global_step} loss={float(loss.detach().cpu()):.6f}"
                )

        accelerator.print(
            f"[BT] finished epoch={epoch_idx + 1}/{config.num_epochs} batches={batch_count} global_step={global_step}"
        )

        if eval_batches is not None:
            eval_loss = evaluate_loss(model, eval_batches, accelerator.device)
            accelerator.print(
                f"[BT] validation epoch={epoch_idx + 1}/{config.num_epochs} eval_loss={eval_loss:.6f}"
            )
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
    return TrainingResult(
        global_step=global_step,
        best_eval_loss=best_eval_loss,
        artifact_path=artifact_path,
    )


def evaluate_loss(
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


def predict_batches(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    logits = []
    score_predictions = []
    difficulty_predictions = []
    member_logits = []
    probability_variances = []
    mutual_information = []
    labels = []
    uncertainty_method = getattr(model, "uq_method", "none")

    with torch.no_grad():
        for batch in batches:
            moved_batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=moved_batch["input_ids"],
                attention_mask=moved_batch.get("attention_mask"),
            )
            batch_logits = outputs.logits_cls.detach().float().cpu()
            logits.append(batch_logits.numpy())
            score_predictions.append(outputs.score_pred.detach().float().cpu().numpy())
            difficulty_predictions.append(outputs.difficulty_pred.detach().float().cpu().numpy())
            if outputs.member_logits is not None:
                member_logits.append(outputs.member_logits.detach().float().cpu().numpy())
                probability_variances.append(
                    outputs.probability_variance.detach().float().cpu().numpy()
                )
                mutual_information.append(
                    outputs.mutual_information.detach().float().cpu().numpy()
                )
            if "labels" in batch:
                labels.append(batch["labels"].detach().float().cpu().numpy())

    result = {
        "logits": np.concatenate(logits, axis=0),
        "score_predictions": np.concatenate(score_predictions, axis=0),
        "difficulty_predictions": np.concatenate(difficulty_predictions, axis=0),
        "uncertainty_method": uncertainty_method,
    }
    if member_logits:
        result["member_logits"] = np.concatenate(member_logits, axis=0)
        result["probability_variance"] = np.concatenate(probability_variances, axis=0)
        result["mutual_information"] = np.concatenate(mutual_information, axis=0)
    if labels:
        result["labels"] = np.concatenate(labels, axis=0)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ModernBERT UQ from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of training data to use (0, 1]. Overrides config data.fraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional data root override for config paths that start with data/.",
    )
    return parser.parse_args()


def _resolve_data_path(path: str | None, data_root: Path | None) -> str | None:
    if path is None or data_root is None:
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    parts = candidate.parts
    if parts and parts[0] == "data":
        return str(data_root / Path(*parts[1:]))
    return str(candidate)


def _fraction_output_dir(base_output_dir: str, fraction: float) -> str:
    output_path = Path(base_output_dir)
    frac_label = f"frac_{fraction:.2f}"
    if output_path.name.startswith("frac_"):
        return str(output_path.with_name(frac_label))
    return str(output_path / frac_label)


def build_dataloader(
    path: str | Path | None,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    target_column: str,
    fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader | None, list[dict[str, object]]]:
    if not path:
        return None, []
    records = load_records(path, target_column=target_column)
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        n = max(1, int(len(records) * fraction))
        indices = sorted(rng.choice(len(records), size=n, replace=False).tolist())
        records = [records[i] for i in indices]
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
        collate_fn=UQBatchCollator(),
    )
    return dataloader, records


def build_pair_dataloader(
    path: str | Path | None,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    fraction: float = 1.0,
    seed: int = 42,
) -> DataLoader | None:
    if not path:
        return None
    if fraction < 1.0:
        records = load_pair_records(path)
        rng = np.random.default_rng(seed)
        n = max(1, int(len(records) * fraction))
        indices = sorted(rng.choice(len(records), size=n, replace=False).tolist())
        records = [records[i] for i in indices]
        tokenized = tokenize_pair_records(records, tokenizer=tokenizer, max_length=max_length)
        dataset = PairDataset(tokenized)
        collate_fn = PairBatchCollator()
        dataloader_shuffle = shuffle
    else:
        dataset = PairCsvIterableDataset(path)
        collate_fn = PairTextBatchCollator(tokenizer=tokenizer, max_length=max_length)
        dataloader_shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        collate_fn=collate_fn,
    )


def save_predictions_and_metrics(
    split_name: str,
    records: list[dict[str, object]],
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    score_task_type: str,
    score_target_column: str,
) -> None:
    if not records:
        return
    ids = [str(record.get("id", f"{split_name}_{index}")) for index, record in enumerate(records)]
    labels = np.asarray([record[score_target_column] for record in records], dtype=float)
    difficulty_targets = np.asarray([record["difficulty"] for record in records], dtype=float)

    table = export_prediction_table(
        ids=ids,
        raw_score_outputs=predictions["logits"],
        score_predictions=predictions["score_predictions"],
        predicted_difficulty=predictions["difficulty_predictions"],
        labels=labels,
        score_task_type=score_task_type,
    )
    table["gold_difficulty"] = difficulty_targets
    if "accuracy_score" in records[0]:
        table["accuracy_score"] = np.asarray([record["accuracy_score"] for record in records], dtype=float)
    table.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    metrics = evaluate_predictions(
        labels=labels,
        score_predictions=predictions["score_predictions"],
        raw_score_outputs=predictions["logits"],
        difficulty_predictions=predictions["difficulty_predictions"],
        difficulty_targets=difficulty_targets,
        score_task_type=score_task_type,
    )
    (output_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def save_mc_dropout_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader | None,
    records: list[dict[str, object]],
    split_name: str,
    output_dir: Path,
    num_passes: int,
    score_task_type: str,
    score_target_column: str,
) -> None:
    if dataloader is None or not records or num_passes <= 0:
        return
    device = next(model.parameters()).device
    rows = []
    offset = 0
    for batch in dataloader:
        moved_batch = move_batch_to_device(batch, device)
        mc_outputs = mc_dropout_predict(model=model, batch=moved_batch, num_passes=num_passes)
        batch_size = moved_batch["input_ids"].shape[0]
        for index in range(batch_size):
            record = records[offset + index]
            rows.append(
                {
                    "id": record.get("id", f"{split_name}_{offset + index}"),
                    score_target_column: record[score_target_column],
                    "gold_difficulty": record["difficulty"],
                    "mean_score_prediction": float(
                        mc_outputs["mean_score_prediction"][index].detach().cpu()
                    ),
                    "score_variance": float(
                        mc_outputs["score_variance"][index].detach().cpu()
                    ),
                    "probability_variance": float(
                        mc_outputs["probability_variance"][index].detach().cpu()
                    ),
                    "predictive_entropy": float(
                        mc_outputs["predictive_entropy"][index].detach().cpu()
                    ),
                    "mutual_information": float(
                        mc_outputs["mutual_information"][index].detach().cpu()
                    ),
                    "mean_difficulty": float(mc_outputs["mean_difficulty"][index].detach().cpu()),
                    "difficulty_variance": float(
                        mc_outputs["difficulty_variance"][index].detach().cpu()
                    ),
                }
            )
        offset += batch_size
    artifact_dir = ensure_directory(output_dir / "mc_dropout")
    import pandas as pd

    pd.DataFrame(rows).to_csv(artifact_dir / f"{split_name}_mc_dropout_predictions.csv", index=False)


def run_experiment(config: ExperimentConfig) -> TrainingResult:
    from transformers import AutoTokenizer

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = ensure_directory(config.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    loss_type = getattr(config, "loss_type", "mse")

    data_fraction = getattr(config, "data_fraction", 1.0)
    train_loader = None
    _train_records: list[dict[str, object]] = []
    if loss_type not in {"bt", "bt_margin"}:
        train_loader, _train_records = build_dataloader(
            config.train_path,
            tokenizer=tokenizer,
            max_length=config.max_length,
            batch_size=config.train_batch_size,
            shuffle=True,
            target_column=config.score_target_column,
            fraction=data_fraction,
            seed=config.seed,
        )
    validation_loader, validation_records = build_dataloader(
        config.validation_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.eval_batch_size,
        shuffle=False,
        target_column=config.score_target_column,
    )
    test_loader, test_records = build_dataloader(
        config.test_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.eval_batch_size,
        shuffle=False,
        target_column=config.score_target_column,
    )

    model = ModernBertUQModel(
        model_name=config.model_name,
        dropout_prob=config.dropout_prob,
        score_task_type=config.score_task_type,
        score_loss_weight=config.score_loss_weight,
        difficulty_loss_weight=config.difficulty_loss_weight,
        positive_class_weight=config.positive_class_weight,
        negative_class_weight=config.negative_class_weight,
        pooling_strategy=config.pooling_strategy,
        uq_method=config.uq_method,
        num_ensemble_members=config.num_ensemble_members,
    )

    center_rewards_coeff = getattr(config, "center_rewards_coeff", 0.0)

    if loss_type in {"bt", "bt_margin"}:
        pair_train_loader = build_pair_dataloader(
            config.train_path,
            tokenizer=tokenizer,
            max_length=config.max_length,
            batch_size=config.train_batch_size,
            shuffle=True,
            fraction=data_fraction,
            seed=config.seed,
        )
        if pair_train_loader is None:
            raise ValueError("BT training requires train_path pointing to a pair CSV")
        result = run_bt_training(
            model=model,
            train_pair_batches=pair_train_loader,
            eval_batches=validation_loader,
            config=TrainConfig(
                learning_rate=config.learning_rate,
                num_epochs=config.num_epochs,
                output_dir=output_dir,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision=config.mixed_precision,
                weight_decay=config.weight_decay,
            ),
            use_margin=(loss_type == "bt_margin"),
            center_rewards_coeff=center_rewards_coeff,
        )
    else:
        if train_loader is None:
            raise ValueError("Training data is required")
        result = run_training(
            model=model,
            train_batches=list(train_loader),
            eval_batches=list(validation_loader) if validation_loader is not None else None,
            config=TrainConfig(
                learning_rate=config.learning_rate,
                num_epochs=config.num_epochs,
                output_dir=output_dir,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision=config.mixed_precision,
                weight_decay=config.weight_decay,
            ),
        )

    if validation_loader is not None:
        validation_predictions = predict_batches(model, list(validation_loader))
        save_predictions_and_metrics(
            split_name="validation",
            records=validation_records,
            predictions=validation_predictions,
            output_dir=output_dir,
            score_task_type=config.score_task_type,
            score_target_column=config.score_target_column,
        )
        save_mc_dropout_predictions(
            model=model,
            dataloader=validation_loader,
            records=validation_records,
            split_name="validation",
            output_dir=output_dir,
            num_passes=config.mc_dropout_passes,
            score_task_type=config.score_task_type,
            score_target_column=config.score_target_column,
        )
    if test_loader is not None:
        test_predictions = predict_batches(model, list(test_loader))
        save_predictions_and_metrics(
            split_name="test",
            records=test_records,
            predictions=test_predictions,
            output_dir=output_dir,
            score_task_type=config.score_task_type,
            score_target_column=config.score_target_column,
        )
        save_mc_dropout_predictions(
            model=model,
            dataloader=test_loader,
            records=test_records,
            split_name="test",
            output_dir=output_dir,
            num_passes=config.mc_dropout_passes,
            score_task_type=config.score_task_type,
            score_target_column=config.score_target_column,
        )

    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.fraction is not None:
        import dataclasses
        config = dataclasses.replace(config, data_fraction=args.fraction)
        config = dataclasses.replace(config, output_dir=_fraction_output_dir(config.output_dir, args.fraction))
    if args.output_dir is not None:
        import dataclasses
        config = dataclasses.replace(config, output_dir=args.output_dir)
    data_root = args.data_root
    if data_root is None:
        import os
        if os.environ.get("DATA_ROOT"):
            data_root = Path(os.environ["DATA_ROOT"])
    if data_root is not None:
        import dataclasses
        config = dataclasses.replace(
            config,
            train_path=_resolve_data_path(config.train_path, data_root),
            validation_path=_resolve_data_path(config.validation_path, data_root),
            test_path=_resolve_data_path(config.test_path, data_root),
        )
    run_experiment(config)


if __name__ == "__main__":
    main()
