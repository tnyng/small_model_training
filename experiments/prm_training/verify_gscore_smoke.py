"""Fail closed before full training if a matched 20-CoT GPU smoke is invalid."""
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
objective = sys.argv[2]
report = json.loads((root / "metrics.json").read_text())
assert report["config"].get("g_objective", "ce") == objective
assert report["epoch"] == 0
for partition in ("train", "dev"):
    coverage = report["coverage"][partition]
    assert coverage["input_cots"] == coverage["one_pass_cots"] == 20
    assert not coverage["excluded"]
predictions = [json.loads(line) for line in (root / "dev_predictions_epoch0.jsonl").read_text().splitlines()]
assert len(predictions) == report["coverage"]["dev"]["input_sentences"]
assert len({r["uid"] for r in predictions}) == len(predictions)
assert len({r["cot_id"] for r in predictions}) == 20
for row in predictions:
    scores = row["scores"]
    assert all(math.isfinite(scores[k]) for k in ("G5", "G_expected", "F"))
    assert 0 <= scores["G5"] <= 1 and 0 <= scores["F"] <= 1
    assert 1 <= scores["G_expected"] <= 5.000001
    if objective == "ce":
        assert len(scores["G_distribution"]) == 5
        assert abs(sum(scores["G_distribution"]) - 1) < 1e-5
    else:
        assert scores["G_distribution"] is None
        assert row["G5_score_kind"] == "normalized_regression_not_probability"
for head in ("G", "F"):
    assert sum(bool(r["masks"][head]) for r in predictions) == report["results"][head]["n"]
assert all(math.isfinite(report["results"]["G"][key]) for key in ("mae_expected", "mse_expected"))
assert (root / "checkpoint_best_g5.pt").is_file()
print(f"SMOKE PASSED: {root.name}, {len(predictions)} aligned predictions, objective={objective}")
