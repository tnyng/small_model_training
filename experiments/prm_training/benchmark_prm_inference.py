"""Matched GPU benchmark: score every sentence in each completed dev CoT.

Both models use the same CoTs, hardware, tokenizer family and pre-tokenized inputs.
Timing excludes checkpoint loading and CPU tokenization, which are reported separately.
E2a batches a CoT's sentences; whole-CoT encodes that CoT once.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from train_prm import PRM, Rows
from train_wholecot_prm import WholeCotDataset, WholeCotPRM
from wholecot_data import load_cots


def timed_inference(cots, encoded, model, mode, device, chunk_size):
    latency = []
    scored = 0
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        for cot, item in zip(cots, encoded, strict=True):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if mode == "e2a":
                for pos in range(0, len(item), chunk_size):
                    chunk = item[pos:pos + chunk_size]
                    ids = torch.stack([r["input_ids"] for r in chunk]).to(device)
                    mask = torch.stack([r["attention_mask"] for r in chunk]).to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits = model(ids, mask)
                    scored += logits["G"].numel()
            else:
                ids = torch.tensor(item["input_ids"], device=device).unsqueeze(0)
                mask = torch.ones_like(ids)
                owner = torch.tensor(item["token_owner"], device=device).unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(ids, mask, owner, len(cot["rows"]))
                scored += logits["G"].numel()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency.append(time.perf_counter() - start)
    return {
        "cots": len(cots), "sentences": scored, "gpu_inference_seconds": sum(latency),
        "cots_per_second": len(cots) / sum(latency),
        "sentences_per_second": scored / sum(latency),
        "latency_p50_ms": float(np.percentile(latency, 50) * 1000),
        "latency_p95_ms": float(np.percentile(latency, 95) * 1000),
        "cuda_peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "mean_sentences_per_cot": statistics.mean(len(cot["rows"]) for cot in cots),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--e2a-checkpoint", type=Path, required=True)
    ap.add_argument("--whole-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--max-cots", type=int, default=0, help="Use a small prefix for smoke tests")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Matched performance benchmark requires an Euler GPU")
    name = "answerdotai/ModernBERT-base"
    cots = load_cots(args.splits / "dev.jsonl")
    if args.max_cots:
        cots = cots[:args.max_cots]
    tok = AutoTokenizer.from_pretrained(name)

    t0 = time.perf_counter()
    e2a_rows = Rows([row for cot in cots for row in cot["rows"]], tok, 512, "sentence_source", ["G", "F"])
    old_encoded = []
    offset = 0
    for cot in cots:
        old_encoded.append([e2a_rows[i] for i in range(offset, offset + len(cot["rows"]))])
        offset += len(cot["rows"])
    old_tokenization = time.perf_counter() - t0
    old_model = PRM(name, ["G", "F"]).to(device)
    old_model.load_state_dict(torch.load(args.e2a_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
    # Warm both implementations before their timed pass.
    timed_inference(cots[:3], old_encoded[:3], old_model, "e2a", device, args.chunk_size)
    old = timed_inference(cots, old_encoded, old_model, "e2a", device, args.chunk_size)
    old["cpu_tokenization_seconds"] = old_tokenization
    old["padded_input_tokens"] = sum(len(item) * 512 for item in old_encoded)
    print(f"[e2a] {old}", flush=True)
    del old_model, old_encoded, e2a_rows
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    whole_data = WholeCotDataset(cots, tok, 8192)
    whole_tokenization = time.perf_counter() - t0
    if len(whole_data) != len(cots):
        raise ValueError("Dev CoT would be excluded, so benchmark inputs would differ")
    whole_model = WholeCotPRM(name).to(device)
    whole_model.load_state_dict(torch.load(args.whole_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
    timed_inference(cots[:3], whole_data.items[:3], whole_model, "whole", device, args.chunk_size)
    whole = timed_inference(cots, whole_data.items, whole_model, "whole", device, args.chunk_size)
    whole["cpu_tokenization_seconds"] = whole_tokenization
    whole["unpadded_input_tokens"] = sum(len(item["input_ids"]) for item in whole_data.items)
    print(f"[whole] {whole}", flush=True)

    result = {"device": torch.cuda.get_device_name(), "e2a": old, "wholecot": whole,
              "inference_speedup_cots_per_second": whole["cots_per_second"] / old["cots_per_second"],
              "note": "Completed-CoT scoring, no cross-CoT batching; CPU tokenization and checkpoint loading excluded from GPU inference timing."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
