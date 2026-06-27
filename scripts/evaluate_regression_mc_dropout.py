from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modernbert_uq.calibration import binary_log_loss
from modernbert_uq.evaluate import expected_calibration_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MC dropout calibration for regression outputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Model output directory.")
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=0.875,
        help="Binary calibration reference threshold on score_target. Default corresponds to accuracy_score >= 8.",
    )
    return parser.parse_args()


def load_binary_reference(frame: pd.DataFrame, threshold: float) -> pd.Series:
    if "score_target" in frame.columns:
        return (frame["score_target"].astype(float) >= threshold).astype(int)
    raise ValueError("MC dropout predictions must contain score_target")


def metric_bundle(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    return {
        "ece": float(expected_calibration_error(labels.to_numpy(), scores.to_numpy())),
        "brier_score": float(((scores - labels) ** 2).mean()),
        "log_loss": float(binary_log_loss(labels.to_numpy(), scores.to_numpy())),
        "mean_score_prediction": float(scores.mean()),
        "mean_score_variance": float(scores.var()),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    mc_dir = output_dir / "mc_dropout"

    validation = pd.read_csv(mc_dir / "validation_mc_dropout_predictions.csv")
    test = pd.read_csv(mc_dir / "test_mc_dropout_predictions.csv")

    val_labels = load_binary_reference(validation, threshold=args.reference_threshold)
    test_labels = load_binary_reference(test, threshold=args.reference_threshold)

    metrics = {
        "reference_threshold": args.reference_threshold,
        "validation": metric_bundle(val_labels, validation["mean_score_prediction"]),
        "test": metric_bundle(test_labels, test["mean_score_prediction"]),
    }

    (mc_dir / "mc_dropout_calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote MC dropout calibration metrics to {mc_dir}")


if __name__ == "__main__":
    main()
