from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize regression experiment outputs.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory containing regression experiment output folders.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="rulefollower_regression",
        help="Prefix for summary artifact filenames.",
    )
    return parser.parse_args()


def parse_name(output_name: str) -> tuple[str, str, str]:
    prefix = "rulefollower_"
    if not output_name.startswith(prefix):
        raise ValueError(f"Unexpected output directory name: {output_name}")
    body = output_name[len(prefix) :]

    variant_markers = [
        "_reg_base",
        "_reg_down_exact",
        "_reg_over_exact",
        "_reg_down_bins",
        "_reg_over_bins",
    ]
    for marker in variant_markers:
        if body.endswith(marker):
            branch_stub = body[: -len(marker)]
            variant = marker[1:]
            break
    else:
        raise ValueError(f"Could not parse variant from {output_name}")

    parser_name, annotator = branch_stub.split("_ann_", maxsplit=1)
    return parser_name, annotator, variant


def load_metrics(metrics_path: Path) -> dict[str, float]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = payload["score_regression"]
    difficulty = payload["difficulty"]
    return {
        "score_rmse": float(score["rmse"]),
        "score_mae": float(score["mae"]),
        "score_pearson": float(score["pearson"]),
        "score_spearman": float(score["spearman"]),
        "difficulty_rmse": float(difficulty["rmse"]),
        "difficulty_mae": float(difficulty["mae"]),
        "difficulty_pearson": float(difficulty["pearson"]),
        "difficulty_spearman": float(difficulty["spearman"]),
    }


def load_kendall_metrics(predictions_path: Path) -> dict[str, float]:
    frame = pd.read_csv(predictions_path)
    metrics: dict[str, float] = {}

    if "score_target" in frame.columns and "score_prediction" in frame.columns:
        metrics["score_kendall_tau"] = float(
            frame["score_target"].corr(frame["score_prediction"], method="kendall")
        )

    if "gold_difficulty" in frame.columns and "predicted_difficulty" in frame.columns:
        metrics["difficulty_kendall_tau"] = float(
            frame["gold_difficulty"].corr(frame["predicted_difficulty"], method="kendall")
        )

    return metrics


def collect_rows(outputs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for output_dir in sorted(outputs_root.glob("rulefollower_*_reg_*")):
        test_metrics = output_dir / "test_metrics.json"
        validation_metrics = output_dir / "validation_metrics.json"
        validation_predictions = output_dir / "validation_predictions.csv"
        test_predictions = output_dir / "test_predictions.csv"
        if (
            not test_metrics.exists()
            or not validation_metrics.exists()
            or not validation_predictions.exists()
            or not test_predictions.exists()
        ):
            continue

        parser_name, annotator, variant = parse_name(output_dir.name)
        row: dict[str, object] = {
            "output_dir": output_dir.name,
            "parser": parser_name,
            "annotator": annotator,
            "variant": variant,
        }
        row.update({f"validation_{k}": v for k, v in load_metrics(validation_metrics).items()})
        row.update({f"test_{k}": v for k, v in load_metrics(test_metrics).items()})
        row.update(
            {f"validation_{k}": v for k, v in load_kendall_metrics(validation_predictions).items()}
        )
        row.update({f"test_{k}": v for k, v in load_kendall_metrics(test_predictions).items()})
        rows.append(row)
    return rows


def reorder_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "output_dir",
        "parser",
        "annotator",
        "variant",
        "validation_score_rmse",
        "validation_score_mae",
        "validation_score_pearson",
        "validation_score_spearman",
        "validation_score_kendall_tau",
        "validation_difficulty_rmse",
        "validation_difficulty_mae",
        "validation_difficulty_pearson",
        "validation_difficulty_spearman",
        "validation_difficulty_kendall_tau",
        "test_score_rmse",
        "test_score_mae",
        "test_score_pearson",
        "test_score_spearman",
        "test_score_kendall_tau",
        "test_difficulty_rmse",
        "test_difficulty_mae",
        "test_difficulty_pearson",
        "test_difficulty_spearman",
        "test_difficulty_kendall_tau",
    ]
    existing = [column for column in preferred if column in frame.columns]
    remaining = [column for column in frame.columns if column not in existing]
    return frame[existing + remaining]


def build_rebalancing_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    comparison_rows: list[dict[str, object]] = []
    for (parser_name, annotator), sub in frame.groupby(["parser", "annotator"], sort=True):
        baseline = sub.loc[sub["variant"] == "reg_base"]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        for _, row in sub.sort_values("variant").iterrows():
            comparison_rows.append(
                {
                    "parser": parser_name,
                    "annotator": annotator,
                    "variant": row["variant"],
                    "test_score_pearson": row["test_score_pearson"],
                    "delta_test_score_pearson_vs_base": row["test_score_pearson"]
                    - baseline_row["test_score_pearson"],
                    "test_score_rmse": row["test_score_rmse"],
                    "delta_test_score_rmse_vs_base": row["test_score_rmse"]
                    - baseline_row["test_score_rmse"],
                    "test_score_kendall_tau": row["test_score_kendall_tau"],
                    "delta_test_score_kendall_tau_vs_base": row["test_score_kendall_tau"]
                    - baseline_row["test_score_kendall_tau"],
                    "test_difficulty_pearson": row["test_difficulty_pearson"],
                    "delta_test_difficulty_pearson_vs_base": row["test_difficulty_pearson"]
                    - baseline_row["test_difficulty_pearson"],
                    "test_difficulty_rmse": row["test_difficulty_rmse"],
                    "delta_test_difficulty_rmse_vs_base": row["test_difficulty_rmse"]
                    - baseline_row["test_difficulty_rmse"],
                    "test_difficulty_kendall_tau": row["test_difficulty_kendall_tau"],
                    "delta_test_difficulty_kendall_tau_vs_base": row[
                        "test_difficulty_kendall_tau"
                    ]
                    - baseline_row["test_difficulty_kendall_tau"],
                }
            )
    return pd.DataFrame(comparison_rows)


def select_top_variants(frame: pd.DataFrame) -> pd.DataFrame:
    selections: list[pd.Series] = []
    for (parser_name, annotator), sub in frame.groupby(["parser", "annotator"], sort=True):
        ranked = sub.sort_values(
            by=["test_score_pearson", "test_score_rmse"],
            ascending=[False, True],
        ).head(2)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            enriched = row.copy()
            enriched["selection_rank"] = rank
            selections.append(enriched)
    if not selections:
        return pd.DataFrame()
    selected = pd.DataFrame(selections)
    selected = selected[
        [
            "selection_rank",
            "output_dir",
            "parser",
            "annotator",
            "variant",
            "test_score_pearson",
            "test_score_rmse",
            "test_score_spearman",
            "test_score_kendall_tau",
            "test_difficulty_pearson",
            "test_difficulty_rmse",
            "test_difficulty_spearman",
            "test_difficulty_kendall_tau",
        ]
    ]
    return selected.sort_values(["parser", "annotator", "selection_rank"])


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".tsv":
        frame.to_csv(path, sep="\t", index=False)
    else:
        frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    rows = collect_rows(outputs_root)
    if not rows:
        raise SystemExit(f"No completed regression outputs found under {outputs_root}")

    summary = reorder_columns(pd.DataFrame(rows).sort_values(["parser", "annotator", "variant"]))
    comparison = build_rebalancing_comparison(summary)
    top_variants = select_top_variants(summary)

    write_table(summary, outputs_root / f"{args.output_prefix}_performance_summary.csv")
    write_table(summary, outputs_root / f"{args.output_prefix}_performance_summary.tsv")
    write_table(comparison, outputs_root / f"{args.output_prefix}_rebalancing_comparison.csv")
    write_table(comparison, outputs_root / f"{args.output_prefix}_rebalancing_comparison.tsv")
    write_table(top_variants, outputs_root / f"{args.output_prefix}_top_variants.csv")
    write_table(top_variants, outputs_root / f"{args.output_prefix}_top_variants.tsv")

    print(f"Wrote regression summaries to {outputs_root}")


if __name__ == "__main__":
    main()
