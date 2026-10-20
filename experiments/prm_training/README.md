# RuleFollower sentence-level ModernBERT experiments

This folder records the experiments **after** the four-generator teacher subset was prepared. It is separate from the older UQ implementation at the repository root. The frozen [Hugging Face subset](https://huggingface.co/datasets/RuleFollower/unified_sentence_gf_annotations) contains 7,932 CoTs and 251,870 numbered sentences from 16 datasets; raw CoTs and binary teacher labels are not duplicated here. G/F results below compare with **automatic teacher labels**, not the still-pending human test labels; E3 and E5 are constructed stress tests. No checkpoints are in this branch.

## Design and progression

The first split groups exact classification-input SHA-256 hashes across all four CoT generators (1,372/297/297 train/dev/teacher-test source groups). The 160-source human pool is excluded. A masked G/F loss ignores technical null labels. The experiments then changed one question at a time:

| Step | What changed and why | Main observation |
|---|---|---|
| TF-IDF and E2b/E2c/E2a | Compare sentence only, +task, then +source to check whether source access helps. These use the original **binary G/F** labels. | Teacher-dev G AP: .780/.788/.816; F AP: .061/.055/.066. The high sentence-only G score warns of shortcuts. |
| E3 source swap; E5 quote edit | Stress-test source sensitivity rather than infer grounding from E2a AP alone. | E3's separately trained source-match AUROC was .977. In E5, the frozen E2a model ranked the correct quote higher in 93.2% of 95 pairs, yet 60% of altered quotes still scored above .5. Neither is a human validation. |
| Whole-CoT binary G/F | Encode the task, source and **complete CoT once**, output both labels for every sentence; target RL scoring efficiency. | Same-A100 GPU inference was 3.63× faster in CoTs/s than E2a (CPU tokenization and model loading excluded). Initial F performance required further work. |
| Longer training and F `pos_weight` | Counter rare F positives, comparing weight 1 and 30 on the same binary task. | Best teacher-dev F AP rose from .0616 to .0732; weight 30 was fixed for the next comparison. |
| G 1–5 score + binary F | Replace only the G teacher target; compare five-class cross-entropy (CE) with normalized-score MSE. F remains binary. | On the fixed teacher test, base CE G=5 AP .8796 / F AP .0523; base MSE .8219 / .0689. MSE's G ranking score is **not** a probability of G=5. The new G labels differ from old binary G, so old-versus-new G differences are not a pure architecture effect. |
| Larger backbone | Keep split, G/F labels, CE objectives, F weight 30, seed 42 and 16-epoch budget; change ModernBERT-base to ModernBERT-large. | Teacher-test G=5 AP .8978 and F AP .0958; F precision among top 20 .50, versus .20 for base CE. Only one training seed has been run. |

The shared whole-CoT model produces a five-class G distribution (or one normalized MSE score) and a binary F score at each sentence position. G=5 means the entire sentence is directly stated or unambiguously supported by the classification input. F=1 means the N1 teacher confirmed an unambiguous fallacy; F=0 includes unflagged cases and is **not** a human-verified nonfallacy. The [G scoring prompt](../../prompt/unified/grounded-score/g1to5_direct_v1.txt) is included for label provenance.

## Analysis and current limit

AP (average precision, a precision–recall summary) is the primary ranking measure because F positives are rare: 196/36,961 valid teacher-test sentences (0.53%). On 297 paired test source groups, large minus base CE AP was +.0181 for G=5 (95% source-bootstrap interval [.0127, .0238]) and +.0435 for F ([.0110, .0783]); large improved G in 14/16 datasets and F in 9/16. These intervals measure source sampling for the **fixed seed**, not training-seed variability or human correctness.

For reward use, a ranking score is insufficient: **precision–coverage** asks how often selected sentences are teacher-positive versus how many valid sentences receive a reward/penalty. The large model's teacher-dev scores were calibrated on one source-disjoint partition, with candidate thresholds selected on another. At an empirical 95% G=5 teacher precision, 1,651/17,608 sentences (9.38% coverage) would be rewarded. No F threshold reached 90% teacher precision with at least 20 selected sentences; calibration improved probability fit but cannot repair ranking. These are exploratory dev operating points, **not deployed reward thresholds**. Human validation and a penalty-policy decision remain pending; no RL results are claimed.

Compact per-run metrics are in [`results_v2/`](results_v2/). The paired source bootstrap, subgroup table, calibration, precision–coverage CSVs and plots are in [`results_v2/backbone_analysis_2026_10_19/analysis/`](results_v2/backbone_analysis_2026_10_19/analysis/). Checkpoints, full split rows, and sentence-level prediction dumps are intentionally omitted. In the published summary JSON files, machine-specific path strings were replaced with logical paths; numeric results were left unchanged.

## Reproduce

1. Download the HF subset into `experiments/hf16_cross_model_full/proportional_2000/hf_upload_teacher_2k_v1/` (preserve its `cots/` and `sentence_labels/` layout). Run `python experiments/prm_training/build_splits_v2.py --human-manifest experiments/prm_training/holdout_source_hashes.csv` from the repository root. The hash-only manifest contains no human labels or source text; [`metadata_split_summary.json`](metadata_split_summary.json) records the expected counts.
2. Decompress `gscore_sidecar_v1/{train,dev,test}.jsonl.gz` beside each compressed file to obtain the G 1–5 sidecar. It is indexed by sentence UID and validated against the binary split at training time. The source teacher's G prompt and sidecar summary are included; the 52 MB uncompressed sidecar is stored as 13 MB gzip files.
3. Run `train_wholecot_prm_gscore.py` with `--splits experiments/prm_training/splits_v2 --g-sidecar experiments/prm_training/gscore_sidecar_v1 --model answerdotai/ModernBERT-base` (or `-large`), `--g-objective ce` (or `mse`), `--f-pos-weight 30 --seed 42 --epochs 16`, and a new `--out` directory. The `.sbatch` files preserve the actual Euler settings but contain site-specific paths. The Python files, tests, and compact per-epoch metrics are retained here.

Dependencies for these experiments include PyTorch, Transformers, NumPy, SciPy, scikit-learn and Matplotlib. The held-out **teacher** test is distinct from the unfinished human test. Do not select new models or thresholds on the teacher-test results above.
