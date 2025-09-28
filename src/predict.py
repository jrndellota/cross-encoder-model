import argparse
import json

import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from utils import concatenate_restaurant_text


def main(split: str, model_path: str):
    restaurants_df = pd.read_parquet("data/restaurants.parquet").set_index(
        "restaurant_id"
    )
    queries_df = pd.read_csv(f"data/queries_{split}.csv").set_index("query_id")
    candidates_df = pd.read_json(
        f"data/bm25_candidates_{split}_top500.jsonl", lines=True
    )

    model = CrossEncoder(f"models/{model_path}")
    output_path = f"runs/{split}_predictions.jsonl"

    print(f"Number of candidate rows: {len(candidates_df)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in tqdm(
            candidates_df.iterrows(),
            total=len(candidates_df),
            desc="Processing queries",
        ):
            query_id = row["query_id"]
            candidates = row["candidates"]
            query = queries_df.loc[query_id, "query_text"]
            pairs = []
            restaurant_ids = []
            for candidate in candidates:
                restaurant_id = candidate["restaurant_id"]
                restaurant_ids.append(restaurant_id)
                r_text = concatenate_restaurant_text(restaurants_df.loc[restaurant_id])
                pairs.append([query, r_text])

            scores = model.predict(pairs, convert_to_numpy=True)
            ranked = sorted(
                zip(restaurant_ids, scores.tolist()), key=lambda x: x[1], reverse=True
            )
            js = {
                "query_id": query_id,
                "candidates": [
                    {"restaurant_id": restaurant_id, "score": float(score)}
                    for restaurant_id, score in ranked
                ],
            }
            f.write(json.dumps(js, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    main(args.split, args.model_path)
