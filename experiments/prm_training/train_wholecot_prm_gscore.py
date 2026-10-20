"""One-pass whole-CoT ModernBERT with a five-class G head and binary F head.

Uses the frozen binary splits plus a separately validated G-score sidecar.
The binary teacher data and the earlier training script are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from train_prm import auc_ap
from train_wholecot_prm import WholeCotDataset, WholeCotPRM, collate
from wholecot_data import load_cots


class GScoreWholeCotPRM(WholeCotPRM):
    def __init__(self, model_name: str, g_objective: str = "ce"):
        super().__init__(model_name)
        self.g_objective = g_objective
        self.heads["G"] = nn.Linear(self.encoder.config.hidden_size, 5 if g_objective == "ce" else 1)


def attach_scores(cots: list[dict], sidecar: Path) -> None:
    by_uid = {}
    with sidecar.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["uid"] in by_uid:
                raise ValueError(f"Duplicate G-score uid: {rec['uid']}")
            by_uid[rec["uid"]] = rec
    used = set()
    for cot in cots:
        for row in cot["rows"]:
            uid = row["uid"]
            label = by_uid.get(uid)
            if label is None:
                raise ValueError(f"Missing G score for {uid}")
            if label["cot_id"] != row["cot_id"] or label["sid"] != row["sid"]:
                raise ValueError(f"G-score ID mismatch: {uid}")
            digest = hashlib.sha256(row["sentence"].encode("utf-8")).hexdigest()
            if label["sentence_sha256"] != digest:
                raise ValueError(f"G-score sentence mismatch: {uid}")
            score = label["G_score"]
            if (score is None) != (not label["mask_G_score"]):
                raise ValueError(f"Inconsistent G-score mask: {uid}")
            if score is not None and (type(score) is not int or score not in range(1, 6)):
                raise ValueError(f"Invalid G score: {uid}")
            row["G_binary"] = row["G"]
            row["mask_G_binary"] = row["mask_G"]
            row["G_score"] = score
            row["G"] = score - 1 if score is not None else None
            row["mask_G"] = score is not None
            used.add(uid)
    if used != set(by_uid):
        raise ValueError(f"Unused sidecar rows: {len(set(by_uid) - used)}")


def quadratic_kappa(y: np.ndarray, pred: np.ndarray) -> float:
    observed = np.zeros((5, 5), dtype=np.float64)
    np.add.at(observed, (y - 1, pred - 1), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16
    denominator = float((weights * expected).sum())
    return float(1 - (weights * observed).sum() / denominator) if denominator else float("nan")


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc, ap = auc_ap(y, p)
    order = np.argsort(-p)
    return {"auc": auc, "ap": ap, "n": len(y),
            "base_rate": float(y.mean()) if len(y) else float("nan"),
            "precision_at_20": float(y[order[:20]].mean()) if len(y) >= 20 else None,
            "precision_at_100": float(y[order[:100]].mean()) if len(y) >= 100 else None}


def loss_for(logits: dict, batch: dict, mean_labels_per_cot: dict,
             f_pos_weight: float, g_objective: str = "ce",
             g_loss_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    g = batch["labels"]["G"].long()
    gm = batch["masks"]["G"]
    f = batch["labels"]["F"]
    fm = batch["masks"]["F"]
    if g_objective == "ce":
        g_per = F.cross_entropy(logits["G"].float().transpose(1, 2), g,
                                reduction="none")
    else:
        # Stored class indices are 0..4; regress their normalized value, not a probability of G=5.
        g_per = (torch.sigmoid(logits["G"].float()) - g.float() / 4).square()
    f_per = F.binary_cross_entropy_with_logits(
        logits["F"].float(), f, reduction="none",
        pos_weight=torch.tensor(f_pos_weight, device=f.device))
    g_loss = (g_per * gm).sum() / (mean_labels_per_cot["G"] * g.shape[0])
    f_loss = (f_per * fm).sum() / (mean_labels_per_cot["F"] * f.shape[0])
    return g_loss_weight * g_loss + f_loss, {"G": g_loss.detach(), "F": f_loss.detach()}


def evaluate(model, loader, device) -> tuple[dict, list[dict]]:
    model.eval()
    gy, gp, fy, fp = [], [], [], []
    rows_out = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            owner = batch["token_owner"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(ids, attention, owner, batch["labels"]["G"].shape[1])
            regression = model.g_objective == "mse"
            probs_g = (torch.sigmoid(logits["G"].float()) if regression else
                       torch.softmax(logits["G"].float(), dim=-1)).cpu().numpy()
            probs_f = torch.sigmoid(logits["F"].float()).cpu().numpy()
            for b, item in enumerate(batch["items"]):
                for s, row in enumerate(item["rows"]):
                    probs = None if regression else [float(v) for v in probs_g[b, s]]
                    g_rank = float(probs_g[b, s]) if regression else probs[4]
                    g_expected = 1 + 4 * g_rank if regression else sum((k + 1) * p for k, p in enumerate(probs))
                    p_f = float(probs_f[b, s])
                    g_valid = bool(row["mask_G"] and row["G_score"] is not None)
                    f_valid = bool(row["mask_F"] and row["F"] is not None)
                    out = {"uid": row["uid"], "cot_id": item["cot_id"],
                           "sid": row["sid"], "generator": row.get("generator"),
                           "task": row.get("task"),
                           "source_text_sha256": row.get("source_text_sha256"),
                           "scores": {"G_distribution": probs, "G5": g_rank,
                                      "G_expected": g_expected,
                                      "F": p_f},
                           "G5_score_kind": "normalized_regression_not_probability" if regression else "class_probability",
                           "labels": {"G_score": row["G_score"],
                                      "G_binary": row["G_binary"], "F": row["F"]},
                           "masks": {"G": g_valid, "F": f_valid}}
                    rows_out.append(out)
                    if g_valid:
                        gy.append(row["G_score"])
                        gp.append(g_rank if regression else probs)
                    if f_valid:
                        fy.append(row["F"])
                        fp.append(p_f)
    y_g = np.asarray(gy, dtype=np.int64)
    p_g = np.asarray(gp, dtype=np.float64)
    expected_g = 1 + 4 * p_g if model.g_objective == "mse" else (p_g * np.arange(1, 6)).sum(axis=1)
    pred_g = np.clip(np.rint(expected_g), 1, 5).astype(int) if model.g_objective == "mse" else p_g.argmax(axis=1) + 1
    g_rank = p_g if model.g_objective == "mse" else p_g[:, 4]
    y_f = np.asarray(fy, dtype=np.int64)
    p_f = np.asarray(fp, dtype=np.float64)
    confusion = np.zeros((5, 5), dtype=int)
    np.add.at(confusion, (y_g - 1, pred_g - 1), 1)
    metrics = {
        "G": {"n": len(y_g), "score_counts": np.bincount(y_g, minlength=6)[1:].tolist(),
              "mae_class": float(np.abs(y_g - pred_g).mean()),
              "mae_expected": float(np.abs(y_g - expected_g).mean()),
              "mse_expected": float(((y_g - expected_g) ** 2).mean()),
              "quadratic_kappa": quadratic_kappa(y_g, pred_g),
              "accuracy": float((y_g == pred_g).mean()),
              "confusion": confusion.tolist(),
              "score5": ranking_metrics((y_g == 5).astype(int), g_rank)},
        "F": ranking_metrics(y_f, p_f),
    }
    return metrics, rows_out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--g-sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--f-pos-weight", type=float, default=30)
    parser.add_argument("--g-objective", choices=("ce", "mse"), default="ce")
    parser.add_argument("--g-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-cots", type=int, default=0)
    parser.add_argument("--max-dev-cots", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--eval-only", choices=("dev", "test"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.f_pos_weight < 1:
        parser.error("--f-pos-weight must be at least 1")
    if not np.isfinite(args.g_loss_weight) or args.g_loss_weight <= 0:
        parser.error("--g-loss-weight must be positive and finite")
    if bool(args.eval_only) != bool(args.checkpoint):
        parser.error("--eval-only and --checkpoint must be used together")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise ValueError("Fast tokenizer required for sentence alignment")
    config = AutoConfig.from_pretrained(args.model)
    if args.max_len > min(config.max_position_embeddings, int(tok.model_max_length)):
        raise ValueError("max_len exceeds model/tokenizer limit")

    partitions = [args.eval_only] if args.eval_only else ["train", "dev"]
    data = {}
    coverage = {}
    for partition in partitions:
        cots = load_cots(args.splits / f"{partition}.jsonl")
        attach_scores(cots, args.g_sidecar / f"{partition}.jsonl")
        if partition == "train" and args.max_train_cots:
            cots = random.Random(args.seed).sample(
                cots, min(args.max_train_cots, len(cots)))
        if partition == "dev" and args.max_dev_cots:
            cots = random.Random(args.seed).sample(cots, min(args.max_dev_cots, len(cots)))
        ds = WholeCotDataset(cots, tok, args.max_len)
        data[partition] = ds
        coverage[partition] = {"input_cots": len(cots), "one_pass_cots": len(ds),
                               "excluded": ds.skipped,
                               "input_sentences": sum(len(c["rows"]) for c in cots)}
        print(f"[coverage] {partition}: {len(ds)}/{len(cots)} CoTs; "
              f"excluded={len(ds.skipped)}", flush=True)
        if not ds:
            raise ValueError(f"No one-pass CoTs remain in {partition}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GScoreWholeCotPRM(args.model, args.g_objective).to(device)
    padding = tok.pad_token_id

    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["config"].get("g_objective", "ce") != args.g_objective:
            raise ValueError("Checkpoint G objective differs from requested objective")
        model.load_state_dict(checkpoint["model_state_dict"])
        loader = DataLoader(data[args.eval_only], batch_size=args.bs,
                            collate_fn=lambda x: collate(x, padding))
        metrics, predictions = evaluate(model, loader, device)
        report = {"partition": args.eval_only, "checkpoint_epoch": checkpoint["epoch"],
                  "results": metrics, "coverage": coverage[args.eval_only]}
        (args.out / f"{args.eval_only}_metrics.json").write_text(json.dumps(report, indent=2))
        write_jsonl(args.out / f"{args.eval_only}_predictions.jsonl", predictions)
        print(f"[eval-only] {args.eval_only} {metrics}", flush=True)
        return

    train_loader = DataLoader(data["train"], batch_size=args.bs, shuffle=True,
                              collate_fn=lambda x: collate(x, padding))
    dev_loader = DataLoader(data["dev"], batch_size=args.bs,
                            collate_fn=lambda x: collate(x, padding))
    updates = ((len(train_loader) + args.grad_accum - 1) // args.grad_accum
               * args.epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * updates), updates)
    means = {h: sum(bool(r.get(f"mask_{h}") and r.get(h) is not None)
                    for item in data["train"].items for r in item["rows"])
             / len(data["train"]) for h in ("G", "F")}
    if any(v == 0 for v in means.values()):
        raise ValueError("A training head has no labelled sentences")
    print(f"[cfg] device={device} train_cots={len(data['train'])} "
          f"dev_cots={len(data['dev'])} updates={updates} "
          f"f_pos_weight={args.f_pos_weight}", flush=True)
    best = {"G5": {"ap": -1., "epoch": None},
            "F": {"ap": -1., "epoch": None}}
    start = time.time()
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            owner = batch["token_owner"].to(device)
            for h in ("G", "F"):
                batch["labels"][h] = batch["labels"][h].to(device)
                batch["masks"][h] = batch["masks"][h].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(ids, attention, owner, batch["labels"]["G"].shape[1])
                loss, components = loss_for(logits, batch, means, args.f_pos_weight,
                                            args.g_objective, args.g_loss_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: epoch={epoch} step={step}")
                loss = loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"[train] epoch={epoch} step={step}/{len(train_loader)} "
                      f"G_loss={components['G'].item():.4f} "
                      f"F_loss={components['F'].item():.4f} "
                      f"elapsed={time.time()-start:.0f}s", flush=True)
        metrics, predictions = evaluate(model, dev_loader, device)
        report = {"epoch": epoch, "config": {k: str(v) for k, v in vars(args).items()},
                  "coverage": coverage, "results": metrics}
        (args.out / f"metrics_epoch{epoch}.json").write_text(json.dumps(report, indent=2))
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
        write_jsonl(args.out / f"dev_predictions_epoch{epoch}.jsonl", predictions)
        for target, ap in (("G5", metrics["G"]["score5"]["ap"]),
                           ("F", metrics["F"]["ap"])):
            if np.isfinite(ap) and ap > best[target]["ap"]:
                best[target] = {"ap": ap, "epoch": epoch}
                if not args.no_checkpoint:
                    torch.save({"model_state_dict": model.state_dict(),
                                "config": vars(args), "epoch": epoch},
                               args.out / f"checkpoint_best_{target.lower()}.pt")
        (args.out / "best_epochs.json").write_text(json.dumps(best, indent=2))
        print(f"[eval] epoch={epoch} G5_AP={metrics['G']['score5']['ap']:.4f} "
              f"F_AP={metrics['F']['ap']:.4f} elapsed={time.time()-start:.0f}s",
              flush=True)
    tok.save_pretrained(args.out / "tokenizer")


if __name__ == "__main__":
    main()
