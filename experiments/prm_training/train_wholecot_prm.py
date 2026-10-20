"""One ModernBERT encoding per complete CoT, with one G/F score per sentence.

This is deliberately separate from train_prm.py (the sentence-wise E2 baseline).
Overlong CoTs are reported and excluded, never silently truncated. The first
experiment therefore measures both model quality and one-pass coverage.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_prm import auc_ap
from wholecot_data import load_cots, make_text, token_sentence_ids


HEADS = ("G", "F")


class WholeCotDataset(Dataset):
    def __init__(self, cots: list[dict], tok, max_len: int):
        self.items = []
        self.skipped = []
        self.lengths = []
        for cot in cots:
            text, spans = make_text(cot)
            enc = tok(text, return_offsets_mapping=True, truncation=False,
                      add_special_tokens=True)
            length = len(enc["input_ids"])
            self.lengths.append(length)
            if length > max_len:
                self.skipped.append({"cot_id": cot["cot_id"], "tokens": length,
                                     "sentences": len(spans)})
                continue
            owner = token_sentence_ids(enc["offset_mapping"], spans)
            present = set(owner)
            if any(j not in present for j in range(len(spans))):
                raise ValueError(f"Sentence lost in token alignment: {cot['cot_id']}")
            self.items.append({
                "cot_id": cot["cot_id"], "rows": cot["rows"],
                "input_ids": enc["input_ids"], "token_owner": owner,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(items: list[dict], pad_id: int) -> dict:
    batch = len(items)
    max_tokens = max(len(x["input_ids"]) for x in items)
    max_sentences = max(len(x["rows"]) for x in items)
    ids = torch.full((batch, max_tokens), pad_id, dtype=torch.long)
    attention = torch.zeros((batch, max_tokens), dtype=torch.long)
    owner = torch.full((batch, max_tokens), -1, dtype=torch.long)
    labels = {h: torch.zeros((batch, max_sentences)) for h in HEADS}
    masks = {h: torch.zeros((batch, max_sentences)) for h in HEADS}
    for b, item in enumerate(items):
        n = len(item["input_ids"])
        ids[b, :n] = torch.tensor(item["input_ids"])
        attention[b, :n] = 1
        owner[b, :n] = torch.tensor(item["token_owner"])
        for s, row in enumerate(item["rows"]):
            for h in HEADS:
                value = row.get(h)
                if value is not None and row.get(f"mask_{h}"):
                    labels[h][b, s] = float(value)
                    masks[h][b, s] = 1
    return {"input_ids": ids, "attention_mask": attention, "token_owner": owner,
            "labels": labels, "masks": masks, "items": items}


class WholeCotPRM(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({h: nn.Linear(hidden, 1) for h in HEADS})

    def forward(self, input_ids, attention_mask, token_owner, n_sentences):
        states = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        batch, _, hidden = states.shape
        valid = token_owner >= 0
        indices = token_owner.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden)
        # Accumulate in float32: bf16 counts lose integer precision on long sentences.
        pooled = torch.zeros((batch, n_sentences, hidden), device=states.device, dtype=torch.float32)
        pooled.scatter_add_(1, indices, states.float() * valid.unsqueeze(-1))
        counts = torch.zeros((batch, n_sentences, 1), device=states.device, dtype=torch.float32)
        counts.scatter_add_(1, token_owner.clamp(min=0).unsqueeze(-1),
                            valid.unsqueeze(-1).to(states.dtype))
        pooled = self.dropout(pooled / counts.clamp(min=1))
        return {h: self.heads[h](pooled).squeeze(-1) for h in HEADS}


def loss_for(logits, batch, mean_labels_per_cot, f_pos_weight: float = 1.0):
    loss = 0.0
    for h in HEADS:
        y = batch["labels"][h]
        mask = batch["masks"][h]
        weight = f_pos_weight if h == "F" else 1.0
        loss_fn = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor(weight, device=logits[h].device)
        )
        # A constant denominator keeps every labelled sentence at equal weight,
        # rather than giving short CoTs the same total weight as long CoTs.
        normalizer = mean_labels_per_cot[h] * y.shape[0]
        loss = loss + (loss_fn(logits[h].float(), y) * mask).sum() / normalizer
    return loss


def evaluate(model, loader, device):
    model.eval()
    values = {h: {"y": [], "p": []} for h in HEADS}
    rows_out = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            owner = batch["token_owner"].to(device)
            n_sentences = batch["labels"]["G"].shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(ids, attention, owner, n_sentences)
            scores = {h: torch.sigmoid(logits[h].float()).cpu().numpy() for h in HEADS}
            for b, item in enumerate(batch["items"]):
                for s, row in enumerate(item["rows"]):
                    out = {"uid": row["uid"], "cot_id": item["cot_id"],
                           "sid": row["sid"], "generator": row.get("generator"),
                           "task": row.get("task"), "source_text_sha256": row.get("source_text_sha256"),
                           "scores": {}, "labels": {}, "masks": {}}
                    for h in HEADS:
                        p = float(scores[h][b, s])
                        mask = bool(row.get(f"mask_{h}") and row.get(h) is not None)
                        out["scores"][h] = p
                        out["labels"][h] = row.get(h)
                        out["masks"][h] = mask
                        if mask:
                            values[h]["p"].append(p)
                            values[h]["y"].append(int(row[h]))
                    rows_out.append(out)
    metrics = {}
    for h in HEADS:
        y = np.array(values[h]["y"])
        p = np.array(values[h]["p"])
        auc, ap = auc_ap(y, p)
        metrics[h] = {"auc": auc, "ap": ap, "n": len(y),
                      "base_rate": float(y.mean()) if len(y) else float("nan")}
    return metrics, rows_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--f-pos-weight", type=float, default=1.0,
                        help="Positive-class loss weight for F only; G stays unweighted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-cots", type=int, default=0)
    parser.add_argument("--max-dev-cots", type=int, default=0)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="For local smoke tests; full runs should save the model")
    parser.add_argument("--checkpoint-policy", choices=("all", "best"), default="all",
                        help="Save every epoch, or only the best G-AP and F-AP checkpoints")
    args = parser.parse_args()
    if args.f_pos_weight < 1:
        parser.error("--f-pos-weight must be at least 1")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise ValueError("Fast tokenizer required for exact token-to-sentence alignment")
    config = AutoConfig.from_pretrained(args.model)
    limit = min(int(config.max_position_embeddings), int(tok.model_max_length))
    if args.max_len > limit:
        raise ValueError(f"max_len={args.max_len} exceeds model/tokenizer limit {limit}")

    train = load_cots(args.splits / "train.jsonl")
    dev = load_cots(args.splits / "dev.jsonl")
    test = load_cots(args.splits / "test.jsonl") if args.profile_only else []
    if args.max_train_cots:
        train = random.Random(args.seed).sample(train, min(args.max_train_cots, len(train)))
    if args.max_dev_cots:
        dev = random.Random(args.seed).sample(dev, min(args.max_dev_cots, len(dev)))
    train_data = WholeCotDataset(train, tok, args.max_len)
    dev_data = WholeCotDataset(dev, tok, args.max_len)
    test_data = WholeCotDataset(test, tok, args.max_len) if args.profile_only else None
    args.out.mkdir(parents=True, exist_ok=True)
    coverage = {}
    partitions = [("train", train_data, train), ("dev", dev_data, dev)]
    if test_data is not None:
        partitions.append(("test", test_data, test))
    for name, data, source_cots in partitions:
        lengths = np.array(data.lengths)
        coverage[name] = {"input_cots": len(lengths), "one_pass_cots": len(data),
                          "excluded_cots": len(data.skipped),
                          "input_sentences": sum(len(c["rows"]) for c in source_cots),
                          "excluded_sentences": sum(x["sentences"] for x in data.skipped),
                          "token_p50": float(np.percentile(lengths, 50)),
                          "token_p90": float(np.percentile(lengths, 90)),
                          "token_p99": float(np.percentile(lengths, 99)),
                          "token_max": int(lengths.max())}
        print(f"[coverage] {name}: {coverage[name]}", flush=True)
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    (args.out / "excluded_overlong.json").write_text(json.dumps({
        "train": train_data.skipped, "dev": dev_data.skipped,
        "test": test_data.skipped if test_data is not None else []}, indent=2))
    if args.profile_only:
        return
    if not train_data or not dev_data:
        raise ValueError("No one-pass CoTs remain in train or dev; lower max_len or change model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WholeCotPRM(args.model).to(device)
    padding = tok.pad_token_id
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True,
                              collate_fn=lambda x: collate(x, padding))
    dev_loader = DataLoader(dev_data, batch_size=args.bs,
                            collate_fn=lambda x: collate(x, padding))
    updates = (len(train_loader) + args.grad_accum - 1) // args.grad_accum * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(opt, int(.1 * updates), updates)
    mean_labels_per_cot = {h: sum(
        bool(r.get(f"mask_{h}") and r.get(h) is not None)
        for item in train_data.items for r in item["rows"]
    ) / len(train_data) for h in HEADS}
    print(f"[cfg] device={device} train_cots={len(train_data)} dev_cots={len(dev_data)} "
          f"batch={args.bs} accumulation={args.grad_accum} updates={updates}", flush=True)
    start = time.time()
    best_ap = {h: float("-inf") for h in HEADS}
    best_epoch = {h: None for h in HEADS}
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            owner = batch["token_owner"].to(device)
            for h in HEADS:
                batch["labels"][h] = batch["labels"][h].to(device)
                batch["masks"][h] = batch["masks"][h].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(ids, attention, owner, batch["labels"]["G"].shape[1])
                loss = loss_for(logits, batch, mean_labels_per_cot,
                                args.f_pos_weight) / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); schedule.step(); opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"[train] epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss_x_accum={loss.item() * args.grad_accum:.4f} elapsed={time.time()-start:.0f}s",
                      flush=True)
        eval_start = time.time()
        metrics, predictions = evaluate(model, dev_loader, device)
        eval_seconds = time.time() - eval_start
        report = {"epoch": epoch, "config": {k: str(v) for k, v in vars(args).items()},
                  "coverage": coverage, "results": metrics,
                  "eval_seconds": eval_seconds,
                  "eval_cots_per_second": len(dev_data) / eval_seconds,
                  "eval_sentences_per_second": len(predictions) / eval_seconds,
                  "cuda_peak_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                          if device.type == "cuda" else None)}
        (args.out / f"metrics_epoch{epoch}.json").write_text(json.dumps(report, indent=2))
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
        with (args.out / f"dev_predictions_epoch{epoch}.jsonl").open("w") as stream:
            for row in predictions:
                stream.write(json.dumps(row) + "\n")
        for h in HEADS:
            ap = metrics[h]["ap"]
            if np.isfinite(ap) and ap > best_ap[h]:
                best_ap[h] = ap
                best_epoch[h] = epoch
                if not args.no_checkpoint and args.checkpoint_policy == "best":
                    torch.save({"model_state_dict": model.state_dict(),
                                "config": vars(args), "epoch": epoch},
                               args.out / f"checkpoint_best_{h.lower()}.pt")
        if not args.no_checkpoint and args.checkpoint_policy == "all":
            torch.save({"model_state_dict": model.state_dict(), "config": vars(args),
                        "epoch": epoch}, args.out / f"checkpoint_epoch{epoch}.pt")
        (args.out / "best_epochs.json").write_text(json.dumps({
            h: {"epoch": best_epoch[h], "ap": best_ap[h]} for h in HEADS
        }, indent=2))
        print(f"[eval] epoch={epoch} {metrics} elapsed={time.time()-start:.0f}s", flush=True)
    tok.save_pretrained(args.out / "tokenizer")


if __name__ == "__main__":
    main()
