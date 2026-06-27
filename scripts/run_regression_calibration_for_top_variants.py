from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temperature scaling and MC-dropout evaluation for selected regression variants."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory containing regression experiment outputs.",
    )
    parser.add_argument(
        "--top-variants",
        type=Path,
        default=None,
        help="Optional CSV/TSV of selected variants. Defaults to <outputs-root>/rulefollower_regression_top_variants.csv",
    )
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=0.875,
        help="Binary reference threshold on normalized score_target for calibration metrics.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def run_script(script: Path, output_dir: Path, threshold: float, repo_root: Path) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--reference-threshold",
            str(threshold),
        ],
        check=True,
        env=env,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = args.outputs_root.resolve()
    top_variants = args.top_variants or (outputs_root / "rulefollower_regression_top_variants.csv")
    selected = read_table(top_variants.resolve())
    selected_dirs = [outputs_root / name for name in selected["output_dir"].tolist()]

    temp_script = repo_root / "scripts" / "calibrate_regression_outputs.py"
    mc_script = repo_root / "scripts" / "evaluate_regression_mc_dropout.py"

    for output_dir in selected_dirs:
        print(f"[RUN] temperature scaling -> {output_dir.name}")
        run_script(temp_script, output_dir=output_dir, threshold=args.reference_threshold, repo_root=repo_root)
        print(f"[RUN] mc-dropout evaluation -> {output_dir.name}")
        run_script(mc_script, output_dir=output_dir, threshold=args.reference_threshold, repo_root=repo_root)

    print(f"[DONE] Calibration artifacts generated for {len(selected_dirs)} selected variants.")


if __name__ == "__main__":
    main()
