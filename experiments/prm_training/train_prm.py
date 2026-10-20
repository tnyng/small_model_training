from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


def auc_ap(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y); s = np.asarray(s)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if len(uniq) < len(s):
        sums = np.zeros(len(uniq)); np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    d = np.argsort(-s); ys = y[d]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
    ap = float((prec * ys).sum() / n_pos)
    return float(auc), ap


class Rows(Dataset):
    def __init__(self, rows, tok, max_len, mode, heads):
        self.rows, self.tok, self.max_len, self.mode, self.heads = rows, tok, max_len, mode, heads

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        first = f"Sentence: {r['sentence']}"
        if self.mode == "sentence_only":
            enc = self.tok(first, truncation=True, max_length=self.max_len, padding="max_length")
        else:
            second = f"Task: {r.get('task_definition','')}"
            if self.mode == "sentence_source":
                second += f"\n\nText: {r.get('classification_input','')}"
            enc = self.tok(first, second, truncation="longest_first", max_length=self.max_len, padding="max_length")
        out = {k: torch.tensor(v) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        for h in self.heads:
            lab = r.get(h)
            out[f"y_{h}"] = torch.tensor(float(lab) if lab is not None else 0.0)
            out[f"m_{h}"] = torch.tensor(float(r.get(f"mask_{h}", lab is not None)))
        return out


class PRM(nn.Module):
    def __init__(self, name, heads, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(h, 1) for k in heads})

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        pooled = self.drop(pooled)
        return {k: v(pooled).squeeze(-1) for k, v in self.heads.items()}


def load(path, tier=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if tier:
        rows = [r for r in rows if r.get("tier") is not None and r["tier"] <= tier]
    return rows


def build_srcswap(rows, seed=0):
    """E3: distinguish original from a different same-task source (capability control)."""
    pos = [r for r in rows if r.get("G") == 1 and r.get("mask_G")]
    by_task = {}
    for r in pos:
        by_task.setdefault(r["task"], []).append(r)
    rng = random.Random(seed); out = []
    for r in pos:
        out.append({**r, "Y": 1, "pair_id": r["uid"]})
        pool = by_task[r["task"]]
        if len(pool) < 2:
            continue
        for _ in range(20):
            o = rng.choice(pool)
            if o.get("source_text_sha256") != r.get("source_text_sha256"):
                out.append({**r, "task_definition": o["task_definition"],
                            "classification_input": o["classification_input"],
                            "Y": 0, "mask_Y": True, "pair_id": r["uid"]})
                break
    for r in out:
        r["mask_Y"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--mode", choices=["sentence_source", "sentence_task", "sentence_only"], default="sentence_source")
    ap.add_argument("--exp", choices=["gf", "srcswap"], default="gf")
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-dev", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    heads = ["Y"] if args.exp == "srcswap" else ["G", "F"]

    tr = load(args.splits / "train.jsonl", args.tier)
    dv = load(args.splits / "dev.jsonl")
    if args.exp == "srcswap":
        tr, dv = build_srcswap(tr, 1), build_srcswap(dv, 2)
    if len(dv) > args.max_dev:
        dv = random.Random(0).sample(dv, args.max_dev)

    tok = AutoTokenizer.from_pretrained(args.model)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRM(args.model, heads).to(dev)

    dl_tr = DataLoader(Rows(tr, tok, args.max_len, args.mode, heads), batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    dl_dv = DataLoader(Rows(dv, tok, args.max_len, args.mode, heads), batch_size=args.bs * 2, num_workers=4)

    steps = len(dl_tr) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] exp={args.exp} mode={args.mode} tier={args.tier} train={len(tr):,} dev={len(dv):,} "
          f"heads={heads} device={dev} steps={steps}", flush=True)
    for h in heads:
        ys = [r[h] for r in tr if r.get(f"mask_{h}") and r.get(h) is not None]
        print(f"[cfg] {h}: {len(ys):,} labelled, {np.mean(ys):.4%} positive", flush=True)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for i, b in enumerate(dl_tr):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(b["input_ids"], b["attention_mask"])
                loss = 0.0
                for h in heads:
                    l = lossf(logits[h].float(), b[f"y_{h}"])
                    m = b[f"m_{h}"]
                    loss = loss + (l * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            if i % 200 == 0:
                print(f"  ep{ep} step {i}/{len(dl_tr)} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

        model.eval(); P = {h: [] for h in heads}; Y = {h: [] for h in heads}; M = {h: [] for h in heads}
        with torch.no_grad():
            for b in dl_dv:
                b = {k: v.to(dev) for k, v in b.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lg = model(b["input_ids"], b["attention_mask"])
                for h in heads:
                    P[h].append(torch.sigmoid(lg[h].float()).cpu().numpy())
                    Y[h].append(b[f"y_{h}"].cpu().numpy()); M[h].append(b[f"m_{h}"].cpu().numpy())
        res = {}
        predictions = {h: np.concatenate(P[h]) for h in heads}
        for h in heads:
            p = np.concatenate(P[h]); y = np.concatenate(Y[h]); m = np.concatenate(M[h]) > 0.5
            a, ap_ = auc_ap(y[m], p[m])
            res[h] = {"auc": a, "ap": ap_, "n": int(m.sum()), "base_rate": float(y[m].mean())}
            if h == "Y":
                paired = {}
                for j, row in enumerate(dv):
                    paired.setdefault(row["pair_id"], {})[int(row["Y"])] = float(p[j])
                usable = [v for v in paired.values() if 0 in v and 1 in v]
                res[h]["paired_accuracy"] = float(np.mean([
                    float(v[1] > v[0]) + 0.5 * float(v[1] == v[0]) for v in usable
                ]))
                res[h]["paired_n"] = len(usable)
            print(f"[eval ep{ep}] {h}: AUC {a:.4f}  AP {ap_:.4f}  (base {y[m].mean():.4%}, n={int(m.sum()):,})", flush=True)
        metrics = {"epoch": ep, "config": {k: str(v) for k, v in vars(args).items()}, "results": res}
        (args.out / f"metrics_epoch{ep}.json").write_text(json.dumps(metrics, indent=2))
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with (args.out / f"dev_predictions_epoch{ep}.jsonl").open("w") as stream:
            for j, row in enumerate(dv):
                stream.write(json.dumps({
                    "uid": row["uid"], "pair_id": row.get("pair_id"),
                    "source_text_sha256": row.get("source_text_sha256"),
                    "generator": row.get("generator"), "task": row.get("task"),
                    "scores": {h: float(predictions[h][j]) for h in heads},
                    "labels": {h: row.get(h) for h in heads},
                    "masks": {h: bool(row.get(f"mask_{h}")) for h in heads},
                }) + "\n")
        torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "epoch": ep},
                   args.out / f"checkpoint_epoch{ep}.pt")
        tok.save_pretrained(args.out / "tokenizer")
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
