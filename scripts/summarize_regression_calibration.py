from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize regression calibration artifacts.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory containing regression experiment outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="rulefollower_regression",
        help="Prefix for calibration summary files.",
    )
    return parser.parse_args()


def parse_name(output_name: str) -> tuple[str, str, str]:
    prefix = "rulefollower_"
    body = output_name[len(prefix) :] if output_name.startswith(prefix) else output_name
    for marker in (
        "_reg_base",
        "_reg_down_exact",
        "_reg_over_exact",
        "_reg_down_bins",
        "_reg_over_bins",
    ):
        if body.endswith(marker):
            branch_stub = body[: -len(marker)]
            variant = marker[1:]
            break
    else:
        raise ValueError(f"Could not parse regression output name: {output_name}")
    parser_name, annotator = branch_stub.split("_ann_", maxsplit=1)
    return parser_name, annotator, variant


def metric_row(output_dir: Path) -> list[dict[str, object]]:
    parser_name, annotator, variant = parse_name(output_dir.name)
    rows: list[dict[str, object]] = []

    temp_metrics = output_dir / "temperature_scaling" / "temperature_scaling_metrics.json"
    if temp_metrics.exists():
        payload = json.loads(temp_metrics.read_text(encoding="utf-8"))
        for split in ("validation", "test"):
            for method, metrics in payload[split].items():
                rows.append(
                    {
                        "output_dir": output_dir.name,
                        "parser": parser_name,
                        "annotator": annotator,
                        "variant": variant,
                        "split": split,
                        "method": "raw" if method == "raw" else "temperature_scaled",
                        "ece": float(metrics["ece"]),
                        "brier_score": float(metrics["brier_score"]),
                        "log_loss": float(metrics["log_loss"]),
                    }
                )

    mc_metrics = output_dir / "mc_dropout" / "mc_dropout_calibration_metrics.json"
    if mc_metrics.exists():
        payload = json.loads(mc_metrics.read_text(encoding="utf-8"))
        for split in ("validation", "test"):
            metrics = payload[split]
            rows.append(
                {
                    "output_dir": output_dir.name,
                    "parser": parser_name,
                    "annotator": annotator,
                    "variant": variant,
                    "split": split,
                    "method": "mc_dropout",
                    "ece": float(metrics["ece"]),
                    "brier_score": float(metrics["brier_score"]),
                    "log_loss": float(metrics["log_loss"]),
                }
            )

    return rows


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".tsv":
        frame.to_csv(path, sep="\t", index=False)
    else:
        frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()

    rows: list[dict[str, object]] = []
    for output_dir in sorted(outputs_root.glob("rulefollower_*_reg_*")):
        rows.extend(metric_row(output_dir))

    if not rows:
        raise SystemExit(f"No calibration artifacts found under {outputs_root}")

    frame = pd.DataFrame(rows).sort_values(
        ["parser", "annotator", "variant", "split", "method"]
    )
    write_table(frame, outputs_root / f"{args.output_prefix}_calibration_summary.csv")
    write_table(frame, outputs_root / f"{args.output_prefix}_calibration_summary.tsv")
    print(f"Wrote calibration summaries to {outputs_root}")


if __name__ == "__main__":
    main()
