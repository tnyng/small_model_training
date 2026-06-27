# LLM Auditor Files for Round 3

This folder contains the deduplicated round-3 audit input for an LLM auditor.

## Files

- `round3_unique_blank.csv`
  - The 53 unique claims seen across `A1_blank.csv` to `A5_blank.csv`
  - Uses the same column schema as the human blank sheets
  - Intended as the direct input sheet for an LLM auditor

- `round3_unique_with_metadata.csv`
  - Same rows as `round3_unique_blank.csv`
  - Adds:
    - `n_human_assignments`
    - `annotator_packets`
  - Useful for analysis, but not necessary to show the model

- `prompt_round3_llm_auditor.txt`
  - Prompt that mirrors the human round-3 rubric
  - Tells the model to fill `claim_well_formed`, `entailment`, `score_correct`, and optional `notes`

- `build_llm_auditor_round3_input.py`
  - Regenerates the two CSV files from the five human packets

## Notes

- The union set should contain 53 unique claims:
  - 47 claims assigned to 3 humans
  - 1 claim assigned to 4 humans
  - 5 singleton overlap-control claims
- `round3_unique_blank.csv` is the safest file to feed to a model because it matches the human sheet format exactly.
