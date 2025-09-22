"""Evaluation utilities for TableCheck re-ranking assignments."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Candidate = Tuple[str, float]


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed qrels line: {line.strip()!r}")
            query_id, doc_id, rel = parts
            qrels[query_id][doc_id] = int(rel)
    return qrels


def load_predictions(path: Path) -> Dict[str, List[Candidate]]:
    predictions: Dict[str, List[Candidate]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            record = json.loads(line)
            query_id = record.get("query_id")
            if not isinstance(query_id, str):
                raise ValueError(f"Line {line_num}: missing query_id")
            if query_id in predictions:
                raise ValueError(f"Duplicate predictions for query_id={query_id}")
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"Line {line_num}: candidates must be a list")
            parsed: List[Candidate] = []
            for cand in candidates:
                if not isinstance(cand, dict):
                    raise ValueError(f"Line {line_num}: candidate entries must be dicts")
                doc_id = cand.get("restaurant_id")
                # Accept either 'score' (preferred) or fallback to 'bm25_score' for BM25 candidate files
                score = cand.get("score", cand.get("bm25_score"))
                if not isinstance(doc_id, str):
                    raise ValueError(f"Line {line_num}: invalid restaurant_id in candidate")
                if not isinstance(score, (int, float)):
                    available_keys = ",".join(sorted(cand.keys()))
                    raise ValueError(
                        f"Line {line_num}: invalid score for restaurant_id={doc_id}; expected numeric 'score' or 'bm25_score'. Keys: {available_keys}"
                    )
                parsed.append((doc_id, float(score)))
            parsed.sort(key=lambda x: x[1], reverse=True)
            predictions[query_id] = parsed
    return predictions


def parse_metric(metric: str) -> Tuple[str, int]:
    if "@" in metric:
        name, _, cutoff = metric.partition("@")
        if not cutoff:
            raise ValueError(f"Invalid metric specifier: {metric}")
        try:
            k = int(cutoff)
        except ValueError as exc:
            raise ValueError(f"Invalid cutoff in metric: {metric}") from exc
    else:
        name, k = metric, 10
    return name.lower(), k


def compute_dcg(ranked: Sequence[Candidate], qrels: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for idx, (doc_id, _) in enumerate(ranked[:k], start=1):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(idx + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[Candidate], qrels: Dict[str, int], k: int) -> float:
    ideal_rels = sorted(qrels.values(), reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal_rels[:k], start=1):
        if rel > 0:
            idcg += (2 ** rel - 1) / math.log2(idx + 1)
    if idcg == 0.0:
        return 0.0
    return compute_dcg(ranked, qrels, k) / idcg


def mrr_at_k(ranked: Sequence[Candidate], qrels: Dict[str, int], k: int) -> float:
    for idx, (doc_id, _) in enumerate(ranked[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / idx
    return 0.0


def recall_at_k(ranked: Sequence[Candidate], qrels: Dict[str, int], k: int) -> float:
    relevant_docs = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant_docs:
        return 0.0
    retrieved = sum(1 for doc_id, _ in ranked[:k] if doc_id in relevant_docs)
    return retrieved / len(relevant_docs)


METRIC_FUNCTIONS = {
    "ndcg": ndcg_at_k,
    "mrr": mrr_at_k,
    "recall": recall_at_k,
}


def evaluate(predictions: Dict[str, List[Candidate]], qrels: Dict[str, Dict[str, int]], metrics: Iterable[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    metric_specs = [parse_metric(metric) for metric in metrics]

    for metric_name, k in metric_specs:
        metric_fn = METRIC_FUNCTIONS.get(metric_name)
        if metric_fn is None:
            raise ValueError(f"Unsupported metric: {metric_name}")
        total = 0.0
        count = 0
        for query_id, query_qrels in qrels.items():
            ranked = predictions.get(query_id, [])
            total += metric_fn(ranked, query_qrels, k)
            count += 1
        results[f"{metric_name}@{k}"] = total / count if count else 0.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate re-ranking predictions against qrels.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to JSONL predictions file.")
    parser.add_argument("--qrels", type=Path, required=True, help="Path to TSV qrels file.")
    parser.add_argument("--metrics", nargs="*", default=["ndcg@10", "mrr@10", "recall@50"], help="List of metrics to compute.")
    parser.add_argument("--output", type=Path, help="Optional path to write metrics as JSON.")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    predictions = load_predictions(args.predictions)
    results = evaluate(predictions, qrels, args.metrics)

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
