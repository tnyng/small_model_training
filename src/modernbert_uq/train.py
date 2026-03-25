from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from accelerate import Accelerator

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
    probabilities = []
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
            batch_logits = outputs.logits_cls.detach().cpu()
            logits.append(batch_logits.numpy())
            if outputs.mean_probability is not None:
                probabilities.append(outputs.mean_probability.detach().cpu().numpy())
            else:
                probabilities.append(torch.sigmoid(batch_logits).numpy())
            difficulty_predictions.append(outputs.difficulty_pred.detach().cpu().numpy())
            if outputs.member_logits is not None:
                member_logits.append(outputs.member_logits.detach().cpu().numpy())
                probability_variances.append(outputs.probability_variance.detach().cpu().numpy())
                mutual_information.append(outputs.mutual_information.detach().cpu().numpy())
            if "labels" in batch:
                labels.append(batch["labels"].detach().cpu().numpy())

    result = {
        "logits": np.concatenate(logits, axis=0),
        "probabilities": np.concatenate(probabilities, axis=0),
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
