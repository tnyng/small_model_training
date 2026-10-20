"""Reconstruct teacher sentence rows as complete, ordered CoTs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


SID = re.compile(r"s([1-9][0-9]*)\Z")


def load_cots(path: Path) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not SID.fullmatch(row["sid"]):
                    raise ValueError(f"Invalid sid: {row['sid']!r}")
                groups[row["cot_id"]].append(row)

    cots = []
    for cot_id, rows in groups.items():
        rows.sort(key=lambda r: int(r["sid"][1:]))
        expected = list(range(1, len(rows) + 1))
        observed = [int(r["sid"][1:]) for r in rows]
        if observed != expected:
            raise ValueError(f"Missing or duplicated sid in {cot_id}: {observed}")
        for field in ("task_definition", "classification_input", "source_text_sha256"):
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f"Inconsistent {field} in {cot_id}")
        cots.append({"cot_id": cot_id, "rows": rows})
    return cots


def make_text(cot: dict) -> tuple[str, list[tuple[int, int]]]:
    rows = cot["rows"]
    first = rows[0]
    text = (f"Task:\n{first['task_definition']}\n\n"
            f"Text:\n{first['classification_input']}\n\nCoT:\n")
    spans = []
    for row in rows:
        text += f"[{row['sid']}] "
        start = len(text)
        text += row["sentence"]
        spans.append((start, len(text)))
        text += "\n"
    return text, spans


def token_sentence_ids(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    """Assign fast-tokenizer offsets to sentence content; exclude headings/IDs."""
    result = [-1] * len(offsets)
    j = 0
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        while j < len(spans) and spans[j][1] <= start:
            j += 1
        if j < len(spans) and start < spans[j][1] and end > spans[j][0]:
            result[i] = j
    return result
