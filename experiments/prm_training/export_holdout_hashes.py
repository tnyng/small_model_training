"""Export only input-text hashes from the private human-pool manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with args.source.open(newline="", encoding="utf-8") as stream:
        hashes = sorted({row["text_sha256"] for row in csv.DictReader(stream)})
    if len(hashes) != 160 or any(len(value) != 64 for value in hashes):
        raise ValueError("Expected 160 distinct SHA-256 input-text hashes")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["text_sha256"])
        writer.writerows((value,) for value in hashes)


if __name__ == "__main__":
    main()
