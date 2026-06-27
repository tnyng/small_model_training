"""Synthesize hard negatives via LLM-edit on a vLLM HTTP server.

Reads positives CSV (row_id, source, claim, split, score, text_id), sends each
through the LLM-edit prompt N times, writes synthetic negatives + failure log.
Parallel HTTP requests via thread pool.
"""
import argparse, csv, json, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

PROMPT = """You are creating a hard negative example for a fact-verification training dataset.

SOURCE: {source}

CLAIM (faithfully entailed by the source): {claim}

Produce a minimally edited version of the claim that:
1. Stays on the same topic as the source.
2. Looks like a real reasoning-step claim (same grammar, same form).
3. Differs from the original claim by 1-3 words ideally.
4. Is NO LONGER entailed by the source -- has a clear factual mismatch.

Output ONLY this JSON, no other text:
{{"edited_claim": "...", "edit_type": "negation|quantifier|stance|number|entity|fact", "why_not_entailed": "<one short sentence>"}}
"""

JSON_RE = re.compile(r"\{[^{}]*\"edited_claim\"[^{}]*\}", re.DOTALL)

def parse_json(text):
    m = JSON_RE.search(text)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

def call(server, model, source, claim, temperature, max_tokens=300, timeout=180):
    p = PROMPT.format(source=source[:1800], claim=claim)
    r = requests.post(f"{server}/v1/chat/completions", json={
        "model": model, "messages":[{"role":"user","content":p}],
        "temperature": temperature, "max_tokens": max_tokens, "seed": 0,
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def process_row(args, row, k_idx, ow, fw, lock, counters):
    temp = args.temperature + 0.1 * k_idx
    try:
        resp = call(args.server, args.model, row["source"], row["claim"], temperature=temp)
    except Exception as e:
        with lock:
            fw.writerow([row["row_id"], row["source"][:200], row["claim"], "", f"http error: {e}"])
            counters["fail"] += 1
        return
    parsed = parse_json(resp)
    if not parsed or "edited_claim" not in parsed:
        with lock:
            fw.writerow([row["row_id"], row["source"][:200], row["claim"], resp[:300], "no JSON"])
            counters["fail"] += 1
        return
    edited = (parsed.get("edited_claim") or "").strip()
    if not edited or edited.lower() == row["claim"].strip().lower():
        with lock:
            fw.writerow([row["row_id"], row["source"][:200], row["claim"], resp[:300], "edit identical to claim"])
            counters["fail"] += 1
        return
    with lock:
        ow.writerow([row["row_id"], row.get("split",""), row["source"], row["claim"], edited,
                     parsed.get("edit_type",""), parsed.get("why_not_entailed",""), temp, k_idx])
        counters["ok"] += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--failed", required=True)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--negatives_per_positive", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0: rows = rows[:args.limit]

    out = open(args.output, "w", newline=""); ow = csv.writer(out)
    ow.writerow(["row_id","split","source","original_claim","edited_claim","edit_type","why_not_entailed","temperature","seed_idx"])
    fail = open(args.failed, "w", newline=""); fw = csv.writer(fail)
    fw.writerow(["row_id","source","claim","raw_response","reason"])

    lock = threading.Lock()
    counters = {"ok":0, "fail":0}
    t0 = time.time()
    futures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for r in rows:
            for k in range(args.negatives_per_positive):
                futures.append(ex.submit(process_row, args, r, k, ow, fw, lock, counters))
        total = len(futures)
        last_print = t0
        for i, fut in enumerate(as_completed(futures)):
            fut.result()
            if time.time() - last_print > 30:
                with lock:
                    rate = (counters["ok"]+counters["fail"]) / max(1, time.time()-t0)
                    eta = (total - (counters["ok"]+counters["fail"])) / max(1, rate)
                    print(f"  [{counters['ok']+counters['fail']}/{total}] ok={counters['ok']} fail={counters['fail']} rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)
                last_print = time.time()
    out.close(); fail.close()
    print(f"Done. ok={counters['ok']} fail={counters['fail']} total_time={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
