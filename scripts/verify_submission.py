"""Lightweight submission format checker for TableCheck assignments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set


def verify_predictions(path: Path) -> int:
    seen_queries: Set[str] = set()
    line_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_num}: invalid JSON") from exc

            query_id = record.get("query_id")
            if not isinstance(query_id, str):
                raise ValueError(f"Line {line_num}: query_id must be a string")
            if query_id in seen_queries:
                raise ValueError(f"Line {line_num}: duplicate query_id {query_id}")
            seen_queries.add(query_id)
            line_count += 1

            candidates = record.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Line {line_num}: candidates must be a non-empty list")

            for idx, cand in enumerate(candidates, start=1):
                if not isinstance(cand, dict):
                    raise ValueError(f"Line {line_num}: candidate #{idx} must be an object")
                doc_id = cand.get("restaurant_id")
                score = cand.get("score")
                if not isinstance(doc_id, str):
                    raise ValueError(f"Line {line_num}: missing restaurant_id for candidate #{idx}")
                if not isinstance(score, (int, float)):
                    raise ValueError(f"Line {line_num}: invalid score for restaurant_id {doc_id}")

    if not seen_queries:
        raise ValueError("No predictions found in file")
    return line_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the structure of a predictions JSONL file.")
    parser.add_argument("predictions", type=Path, help="Path to JSONL file to verify.")
    args = parser.parse_args()

    count = verify_predictions(args.predictions)
    print(f"OK: {args.predictions} contains {count} queries.")


if __name__ == "__main__":
    main()
