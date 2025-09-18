# TableCheck Machine Learning Take-home Assignment

## Take-home Assignment: Restaurant Search Re-ranking

### Scenario & Objective
You are working on TableCheck’s search team. We have an internal corpus of restaurant profiles (name, cuisine tags, location text, free-form descriptions). Customers issue natural-language queries such as “romantic omakase near Ginza” or “kid-friendly brunch in Shibuya with outdoor seating.”  
Your job is to **fine-tune a neural re-ranking model** so that, given a candidate list of restaurants returned by our production BM25 retriever, the re-ranked list places the most relevant restaurants at the top. You must **beat the supplied baseline** on our held-out evaluation set.

### What TableCheck Provides

| Item | Description | File(s) |
| --- | --- | --- |
| Corpus | Restaurant records curated for this exercise. Columns: `restaurant_id`, `name`, `neighborhood`, `cuisines`, `price_tier`, `description`, `tags`. | `data/restaurants.parquet` |
| Queries | Query texts for `train`, `dev`, and `test`. Each row: `query_id`, `query_text`. | `data/queries_train.csv`, `data/queries_dev.csv`, `data/queries_test.csv` |
| Judgements | Relevance labels (0, 1, or 2) for `train`/`dev`. Test labels are held out. | `data/qrels_train.tsv`, `data/qrels_dev.tsv` |
| Candidate lists | Top-500 BM25 candidates per query, used as inputs to your re-ranker. | `data/bm25_candidates_{split}_top500.jsonl` |
| Submission checker | Verifies the format of your prediction file before you submit. | `scripts/verify_submission.py` |

### Requirements

1. **Modeling**
   - Start from any publicly available cross-encoder or dual-encoder transformer (e.g., Sentence-Transformers, Hugging Face).
   - Fine-tune it on the provided training data. You may augment data but must document the process.
   - You may build a two-stage system (new retriever + re-ranker) as long as the final rankings are for the supplied candidate sets.

2. **Evaluation**
   - Use `scripts/evaluate.py` to produce metrics on `dev` during development.
   - Final submission must include rankings for the hidden `test` split in `runs/test_predictions.jsonl` (same structure as candidate files but with your scores).

3. **Deliverables**
   - `src/`: Reproducible training & inference code with instructions (`README.md`).
   - `runs/test_predictions.jsonl`: Your re-ranked lists for the test queries.
   - `models/`: Serialized fine-tuned model or download script.
   - `docs/report.md`: 1–2 pages covering approach, experiments, and discussion of trade-offs.
   - `environment.yml` or `requirements.txt`.

4. **Time Expectation**: 1–4 focused hours. Please note wall-clock spent in your report.

### Scoring & Pass Criteria

We evaluate on the hidden test set using three metrics:

| Metric | Definition | Baseline | Pass Threshold |
| --- | --- | --- | --- |
| `ndcg@10` | Discounted gain at rank 10 | 0.20 | **≥ 0.30** |
| `mrr@10` | Mean reciprocal rank at 10 | 0.20 | **≥ 0.30** |
| `recall@50` | Unique relevant items retrieved in top 50 | 0.50 | **≥ 0.60** |

A submission passes if it beats the thresholds on **at least two** of the three metrics and satisfies the reproducibility checklist.

Secondary evaluation points:

- Code quality & organization
- Experiment methodology (ablation, validation discipline)
- Clarity of the write-up
- Practical considerations (latency estimates, resource usage)

### What you get
- Data files in `data/`: `restaurants.parquet`, `queries_{train,dev,test}.csv`, `qrels_{train,dev}.tsv`, and `bm25_candidates_{train,dev,test}_top500.jsonl`
- A submission format checker: `scripts/verify_submission.py`
- Expected baseline metrics above for comparison

### Submission file format
Submit a JSONL file at `runs/test_predictions.jsonl`. One line per test query:

```json
{"query_id": "q_test_1", "candidates": [{"restaurant_id": "r_odaiba_seafood", "score": 12.34}, {"restaurant_id": "r_shinjuku_rooftop", "score": 11.02}]}
{"query_id": "q_test_2", "candidates": [{"restaurant_id": "r_ginza_tempura", "score": 9.87}, {"restaurant_id": "r_ginza_moon", "score": 8.11}]}
```

- `query_id` must be a string matching an ID in `data/queries_test.csv`.
- `candidates` must be a non-empty list of objects with `restaurant_id` (string) and `score` (number). Higher `score` means better rank.
- Only use `restaurant_id`s from the corresponding `bm25_candidates_test_top500.jsonl` entry for that query.
- Order your `candidates` by descending `score`.

Validate locally:

```bash
python scripts/verify_submission.py runs/test_predictions.jsonl
```

Send your submission as a .zip to TableCheck HR.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `data/` | Corpus, queries, qrels (train/dev), and top-500 candidate lists for all splits. |
| `scripts/` | Submission validation utility (`verify_submission.py`). |

Deliver in `src/` with a reproducible script (train + infer), and include model serialization in `models/` or a download script. Document metrics and trade-offs in `docs/report.md`.
