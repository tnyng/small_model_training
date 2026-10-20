"""E5: score verifiable, one-word corruptions of source-exact quotations.

The cases come only from the machine-held-out test split. This is a diagnostic
for a frozen G model, not new training data or a general paraphrase benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_prm import PRM, Rows


QUOTES = re.compile(r'(["“])([^"”]{15,180})(["”])')
WORD = re.compile(r"\b[A-Za-z]+\b")
OPPOSITES = {
    "good": "bad", "bad": "good", "positive": "negative",
    "negative": "positive", "high": "low", "low": "high",
    "more": "less", "less": "more", "increase": "decrease",
    "decrease": "increase", "increased": "decreased",
    "decreased": "increased", "rising": "falling", "falling": "rising",
    "supports": "opposes", "opposes": "supports",
    "agree": "disagree", "disagree": "agree", "agrees": "disagrees",
    "disagrees": "agrees", "true": "false", "false": "true",
    "legal": "illegal", "illegal": "legal", "allowed": "prohibited",
    "prohibited": "allowed", "safe": "unsafe", "unsafe": "safe",
    "happy": "sad", "sad": "happy", "love": "hate", "hate": "love",
    "loves": "hates", "hates": "loves", "strong": "weak",
    "weak": "strong", "serious": "minor", "minor": "serious",
    "guilty": "innocent", "innocent": "guilty",
}
AUX = {"is", "are", "was", "were"}
QUESTION_SUBJECTS = {"i", "you", "he", "she", "we", "they", "it", "there"}


def preserve_case(old: str, new: str) -> str:
    return new.upper() if old.isupper() else new.capitalize() if old[0].isupper() else new


def corrupt(quote: str, source: str):
    """Return a single material edit, never an altered quote present in source."""
    source_lower = source.casefold()
    candidates = []
    for m in WORD.finditer(quote):
        old = m.group()
        if old.casefold() in OPPOSITES:
            # Negated adjectives can become weak/ambiguous ("not good" -> "not bad").
            if re.search(r"(?:\bnot|n.t)\s+$", quote[max(0, m.start()-10):m.start()], re.I):
                continue
            new = preserve_case(old, OPPOSITES[old.casefold()])
            changed = quote[:m.start()] + new + quote[m.end():]
            candidates.append((changed, "antonym", old, new))
    # An affirmative auxiliary becoming explicitly negative is a clear flip.
    for m in WORD.finditer(quote):
        old = m.group()
        if old.casefold() not in AUX:
            continue
        following = quote[m.end():m.end()+12].lower()
        if re.match(r"\s*(?:not|n't)\b", following):
            continue
        next_word = WORD.search(quote, m.end())
        if next_word and next_word.group().casefold() in QUESTION_SUBJECTS:
            continue
        changed = quote[:m.end()] + " not" + quote[m.end():]
        candidates.append((changed, "negation", old, old + " not"))
    for changed, kind, old, new in candidates:
        if changed.casefold() not in source_lower:
            return changed, kind, old, new
    return None


def make_cases(path: Path, limit: int, seed: int):
    candidates = []
    for line in path.open():
        r = json.loads(line)
        if r.get("G") != 1 or not r.get("mask_G"):
            continue
        source = r["classification_input"]
        for m in QUOTES.finditer(r["sentence"]):
            quote = m.group(2)
            if "?" in quote:
                continue
            if quote.casefold() not in source.casefold():
                continue
            mutation = corrupt(quote, source)
            if mutation is None:
                continue
            changed, kind, old, new = mutation
            altered = r["sentence"][:m.start(2)] + changed + r["sentence"][m.end(2):]
            if altered == r["sentence"]:
                continue
            candidates.append({
                "uid": r["uid"], "source_text_sha256": r["source_text_sha256"],
                "source_key": r["source_key"], "task": r["task"],
                "generator": r["generator"], "task_definition": r["task_definition"],
                "classification_input": source, "sentence_original": r["sentence"],
                "sentence_altered": altered, "quote_original": quote,
                "quote_altered": changed, "edit_type": kind, "edit_old": old,
                "edit_new": new,
            })
            break
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Keep at most one case per source, preventing long CoTs from dominating.
    seen = set()
    cases = []
    for c in candidates:
        if c["source_text_sha256"] in seen:
            continue
        seen.add(c["source_text_sha256"])
        cases.append(c)
        if len(cases) >= limit:
            break
    return cases, len(candidates)


def score(model, tokenizer, cases, device, batch_size, max_len):
    rows = []
    for case in cases:
        for variant in ("original", "altered"):
            rows.append({"sentence": case[f"sentence_{variant}"],
                         "task_definition": case["task_definition"],
                         "classification_input": case["classification_input"],
                         "G": 1 if variant == "original" else 0, "mask_G": True})
    loader = DataLoader(Rows(rows, tokenizer, max_len, "sentence_source", ["G"]),
                        batch_size=batch_size, num_workers=4)
    values = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = model(b["input_ids"].to(device), b["attention_mask"].to(device))["G"]
            values.extend(torch.sigmoid(logit.float()).cpu().tolist())
    for i, case in enumerate(cases):
        case["score_original"] = values[2*i]
        case["score_altered"] = values[2*i+1]


def dev_threshold(path: Path, target_precision: float = .90):
    pairs = []
    for line in path.open():
        row = json.loads(line)
        if row["masks"].get("G") and row["labels"].get("G") is not None:
            pairs.append((float(row["scores"]["G"]), int(row["labels"]["G"])))
    pairs.sort(reverse=True)
    positives = 0
    best = None
    for rank, (score, label) in enumerate(pairs, 1):
        positives += label
        if rank >= 100 and positives / rank >= target_precision:
            best = (score, rank, positives / rank)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--dev-predictions", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases, candidate_count = make_cases(args.test, args.limit, args.seed)
    if not cases:
        raise ValueError("No E5 cases could be constructed")
    print(f"[cases] candidates={candidate_count} unique_source_cases={len(cases)} types={dict(Counter(c['edit_type'] for c in cases))}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PRM(checkpoint["config"]["model"], ["G", "F"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    score(model, tokenizer, cases, device, args.batch_size, args.max_len)
    threshold = dev_threshold(args.dev_predictions)
    orig = np.array([c["score_original"] for c in cases])
    alt = np.array([c["score_altered"] for c in cases])
    summary = {
        "n": len(cases), "candidate_count": candidate_count,
        "test": str(args.test), "checkpoint": str(args.checkpoint),
        "threshold_dev_teacher_precision_90": threshold,
        "pairwise_original_gt_altered": float(np.mean(orig > alt) + .5 * np.mean(orig == alt)),
        "median_score_original": float(np.median(orig)),
        "median_score_altered": float(np.median(alt)),
        "median_paired_drop": float(np.median(orig - alt)),
        "altered_score_ge_0_5": float(np.mean(alt >= .5)),
        "original_score_ge_0_5": float(np.mean(orig >= .5)),
        "by_edit_type": {},
    }
    if threshold:
        t = threshold[0]
        summary["altered_score_ge_dev90_threshold"] = float(np.mean(alt >= t))
        summary["original_score_ge_dev90_threshold"] = float(np.mean(orig >= t))
    for kind in sorted(set(c["edit_type"] for c in cases)):
        subset = [c for c in cases if c["edit_type"] == kind]
        summary["by_edit_type"][kind] = {
            "n": len(subset),
            "pairwise_original_gt_altered": float(np.mean([c["score_original"] > c["score_altered"] for c in subset])),
            "altered_score_ge_0_5": float(np.mean([c["score_altered"] >= .5 for c in subset])),
        }
    with (args.out / "pairs.jsonl").open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
