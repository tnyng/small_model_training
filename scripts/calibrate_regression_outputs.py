from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modernbert_uq.calibration import TemperatureScaler, binary_log_loss
from modernbert_uq.evaluate import expected_calibration_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temperature-scale regression outputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Model output directory.")
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=0.875,
        help="Binary calibration reference threshold on score_target. Default corresponds to accuracy_score >= 8.",
    )
    return parser.parse_args()


def load_labels(frame: pd.DataFrame, threshold: float) -> pd.Series:
    if "score_target" in frame.columns:
        return (frame["score_target"].astype(float) >= threshold).astype(int)
    if "accuracy_score" in frame.columns:
        raw_threshold = (threshold * 8.0) + 1.0
        return (frame["accuracy_score"].astype(float) >= raw_threshold).astype(int)
    raise ValueError("Prediction table must contain score_target or accuracy_score")


def metric_bundle(labels, probabilities, raw_outputs) -> dict[str, float]:
    return {
        "ece": float(expected_calibration_error(labels.to_numpy(), probabilities.to_numpy())),
        "brier_score": float(((probabilities - labels) ** 2).mean()),
        "log_loss": float(binary_log_loss(labels.to_numpy(), probabilities.to_numpy())),
        "mean_raw_output": float(raw_outputs.mean()),
        "mean_probability_like_score": float(probabilities.mean()),
    }


def calibrate_split(frame: pd.DataFrame, scaler: TemperatureScaler, labels: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    out["binary_reference"] = labels
    out["temperature_scaled_score"] = scaler.predict_proba(out["raw_score_output"].to_numpy())
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    calibration_dir = output_dir / "temperature_scaling"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(output_dir / "validation_predictions.csv")
    test = pd.read_csv(output_dir / "test_predictions.csv")

    val_labels = load_labels(validation, threshold=args.reference_threshold)
    test_labels = load_labels(test, threshold=args.reference_threshold)

    scaler = TemperatureScaler().fit(
        logits=validation["raw_score_output"].to_numpy(),
        labels=val_labels.to_numpy(),
    )

    val_calibrated = calibrate_split(validation, scaler=scaler, labels=val_labels)
    test_calibrated = calibrate_split(test, scaler=scaler, labels=test_labels)

    val_calibrated.to_csv(calibration_dir / "validation_temperature_scaled.csv", index=False)
    test_calibrated.to_csv(calibration_dir / "test_temperature_scaled.csv", index=False)

    metrics = {
        "reference_threshold": args.reference_threshold,
        "temperature": scaler.temperature_,
        "validation": {
            "raw": metric_bundle(val_labels, validation["score_prediction"], validation["raw_score_output"]),
            "temperature_scaled": metric_bundle(
                val_labels,
                val_calibrated["temperature_scaled_score"],
                validation["raw_score_output"],
            ),
        },
        "test": {
            "raw": metric_bundle(test_labels, test["score_prediction"], test["raw_score_output"]),
            "temperature_scaled": metric_bundle(
                test_labels,
                test_calibrated["temperature_scaled_score"],
                test["raw_score_output"],
            ),
        },
    }
    (calibration_dir / "temperature_scaling_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote temperature scaling outputs to {calibration_dir}")


if __name__ == "__main__":
    main()
