# Synthetic Negatives — Generation Summary

## What this is

Hard-negative training data for the RFChecker, generated via LLM-edit on the verified positive (claim, source) pairs from the chosen main combo. Each positive (score-8/9 row) is rewritten by an editor LLM into 1–2 minimally-edited claims that are no longer entailed by the source.

## Inputs

| | |
|---|---|
| Main combo (positives source) | `llm gpt-5-mini × gpt-4.1-mini reg_base`, post Stage A.5 filter |
| Positives used | score-8/9 rows from train + validation splits |
| Test split | held out, no synthesis (1 916 positives untouched) |

## Editor model and infrastructure

| | |
|---|---|
| Editor LLM | `google/gemma-4-31B-it` |
| Different from project parsers/annotators? | Yes (project uses gpt-4.1-mini, gpt-5-mini, gpt-oss, Qwen3) |
| Cluster | CSCS Clariden, GH200 GPUs |
| Container | `gemma4_vllm.sqsh` (colleague's, with `patch_vllm_hoist.py` runtime fix) |
| Tensor-parallel | 4 |
| Concurrent client workers | 32 |
| HF cache | `$SCRATCH/hf_home/` (Gemma 4 weights ~59 GB downloaded once) |

## Prompt template (from `synthesize_negatives.py`)

```text
You are creating a hard negative example for a fact-verification training dataset.

SOURCE: {source}

CLAIM (faithfully entailed by the source): {claim}

Produce a minimally edited version of the claim that:
1. Stays on the same topic as the source.
2. Looks like a real reasoning-step claim (same grammar, same form).
3. Differs from the original claim by 1-3 words ideally.
4. Is NO LONGER entailed by the source -- has a clear factual mismatch.

Output ONLY this JSON, no other text:
{"edited_claim": "...", "edit_type": "negation|quantifier|stance|number|entity|fact", "why_not_entailed": "<one short sentence>"}
```

Sampling: temperature 0.3 + 0.1·k (k = 0 or 1 across the two negatives per positive), seed 0, max_tokens 300.

## Results — full run (job 2044345)

| Metric | Value |
|---|---:|
| Positives processed | 11 383 |
| Negatives requested (2 per positive) | 22 766 |
| **Negatives accepted** | **22 154** (97.3%) |
| Failures | 612 (2.7%) |
| Synthesis wall time | 9.5 min |
| Total job time (incl. vLLM load) | 14.5 min |
| Peak throughput | ≈ 39.7 req/s |

### Edit-type distribution

| Type | Count | % |
|---|---:|---:|
| entity (swap) | 6 164 | 27.8 |
| fact (substitution) | 5 784 | 26.1 |
| stance (flip) | 4 949 | 22.3 |
| negation | 3 531 | 15.9 |
| number | 1 226 | 5.5 |
| quantifier | 500 | 2.3 |

### By split

| Split | Positives | Synthetic negatives | Total pairs |
|---|---:|---:|---:|
| train | 9 328 | 18 111 | 27 439 |
| validation | 2 055 | 4 043 | 6 098 |
| test (held out, no synthesis yet) | 1 916 | 0 | — |

### Sanity checks

| Check | Result |
|---|---:|
| Edits identical to original claim | 0 |
| Edits with 0% source-token coverage (off-topic) | 0 / 500 sampled |
| Failure mode | All "no JSON parsed" or "edit identical" — model echoes / refuses |

### Sample edits (random, from `negatives.csv`)

| Original | Edited | Type |
|---|---|---|
| the text quotes Morris describing Priebus reaching out to Latino immigrants | the text quotes **Ryan Mahoney** describing Priebus reaching out to Latino immigrants | entity |
| the text suggests courts are being **asked** to do work that Congress might otherwise do | the text suggests courts are **refusing** to do work that Congress might otherwise do | stance |
| Quetta is described as the headquarters of the exiled Afghan Taliban leadership | **Karachi** is described as the headquarters of the exiled Afghan Taliban leadership | entity |
| numerous **Republicans** considering 2016 presidential campaigns | numerous **Democrats** considering 2016 presidential campaigns | entity |
| Morris was effectively banned … when his contract was **not renewed** | Morris was effectively banned … when his contract was **renewed** | negation |
| Morris **declining** to provide details about when or where the ad might run | Morris **agreeing** to provide details | stance |

All consistent with the spec: 1–3 word minimal edit, on-topic, clearly contradicting source.

## Artifacts on Clariden

```
/iopsstor/scratch/cscs/yufany/RuleFollower/synth/
├── positives_train_val.csv          (input: 11 383 positives, ~25 MB)
├── out_full/
│   ├── negatives.csv                (22 154 synthetic negatives, ~46 MB)
│   └── failed.csv                   (612 failures with reason, ~421 KB)
├── out_proto/
│   ├── negatives.csv                (180 prototype negatives)
│   └── failed.csv                   (20 prototype failures)
├── scripts/
│   ├── synthesize_negatives.py      (parallel HTTP client + prompt template)
│   └── patch_vllm_hoist.py          (runtime patch for the container)
├── run_synth.sbatch                 (SLURM script)
└── logs/run_synth.out               (job log with per-30s progress)
```

## Pending work

1. **Programmatic validators** on the 22 154 negatives:
   - minimum-edit-distance check (number of words changed),
   - source-token coverage (claim still on-topic),
   - optional round-trip annotator score (run `gpt-4.1-mini` on the edited claim — should give 1–3).
2. **Hand-check 30 random negatives** to confirm overall quality (already strong from 8-row spot check).
3. **Build the final RFChecker training corpus**: combine positives + accepted negatives + provenance metadata, save in HF dataset format.
4. **Optional**: run the same pipeline on the held-out test split (1 916 positives → ~3 800 negatives) so you have a balanced evaluation set.

## Reproducibility

To regenerate with the same seed / same model, run on Clariden:

```bash
sbatch --export=ALL,LIMIT=0,NEG=2,TAG=full \
  $SCRATCH/RuleFollower/synth/run_synth.sbatch
```

Inputs and seed (0) are pinned in `synthesize_negatives.py`. Editor model, edit prompt, temperature schedule, and number of workers are documented above and in the script.
