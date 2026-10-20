import json
import tempfile
import unittest
from pathlib import Path

from wholecot_data import load_cots, make_text, token_sentence_ids


class WholeCotDataTest(unittest.TestCase):
    def test_ordering_and_spans(self):
        base = {"cot_id": "one", "task_definition": "Choose a label.",
                "classification_input": "Hello world.", "source_text_sha256": "abc"}
        rows = [{**base, "sid": "s2", "sentence": "Second sentence."},
                {**base, "sid": "s1", "sentence": "First sentence."}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            cots = load_cots(path)
        self.assertEqual([r["sid"] for r in cots[0]["rows"]], ["s1", "s2"])
        text, spans = make_text(cots[0])
        self.assertEqual([text[a:b] for a, b in spans],
                         ["First sentence.", "Second sentence."])
        self.assertEqual(token_sentence_ids([(0, 0), spans[0], spans[1]], spans),
                         [-1, 0, 1])

    def test_missing_sid_is_rejected(self):
        base = {"cot_id": "one", "task_definition": "Task",
                "classification_input": "Text", "source_text_sha256": "abc"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps({**base, "sid": "s2", "sentence": "Oops"}) + "\n")
            with self.assertRaises(ValueError):
                load_cots(path)


if __name__ == "__main__":
    unittest.main()
