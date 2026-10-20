"""Evaluate a frozen binary whole-CoT checkpoint on one teacher split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_wholecot_prm import WholeCotDataset, WholeCotPRM, collate, evaluate
from wholecot_data import load_cots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    tok = AutoTokenizer.from_pretrained(args.model)
    cots = load_cots(args.splits / f"{args.partition}.jsonl")
    data = WholeCotDataset(cots, tok, args.max_len)
    if data.skipped:
        raise ValueError(f"Excluded CoTs from {args.partition}: {data.skipped[:3]}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WholeCotPRM(args.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(data, batch_size=args.bs,
                        collate_fn=lambda x: collate(x, tok.pad_token_id))
    metrics, predictions = evaluate(model, loader, device)
    args.out.mkdir(parents=True)
    report = {"partition": args.partition, "checkpoint": str(args.checkpoint),
              "checkpoint_epoch": checkpoint["epoch"], "results": metrics,
              "cots": len(data), "sentences": len(predictions)}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"epoch": report["checkpoint_epoch"], "results": metrics},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
