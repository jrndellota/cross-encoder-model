from sentence_transformers import CrossEncoder
import pandas as pd
import json
from tqdm import tqdm

restaurants_df = pd.read_parquet("data/restaurants.parquet").set_index("restaurant_id")
queries_dev_df = pd.read_csv("data/queries_dev.csv").set_index("query_id")
candidates_dev_df = pd.read_json("data/bm25_candidates_dev_top500.jsonl", lines=True)

model = CrossEncoder("models/cross-encoder")

output_path = "runs/test_predictions.jsonl"

print(f"Number of candidate rows: {len(candidates_dev_df)}")

with open(output_path, "w", encoding="utf-8") as f:
    for i, row in tqdm(
        candidates_dev_df.iterrows(),
        total=len(candidates_dev_df),
        desc="Processing queries",
    ):
        query_id = row["query_id"]
        candidates = row["candidates"]
        query = queries_dev_df.loc[query_id, "query_text"]
        pairs = []
        restaurant_ids = []
        for candidate in candidates:
            restaurant_id = candidate["restaurant_id"]
            restaurant_ids.append(restaurant_id)
            r_text = (
                "Neighborhood: "
                + restaurants_df.loc[restaurant_id, "neighborhood"]
                + " | Cuisines: "
                + restaurants_df.loc[restaurant_id, "cuisines"]
                + " | Description: "
                + restaurants_df.loc[restaurant_id, "description"]
                + " | Tags: "
                + restaurants_df.loc[restaurant_id, "tags"]
            )
            pairs.append([query, r_text])

        scores = model.predict(pairs, convert_to_numpy=True)
        ranked = sorted(
            zip(restaurant_ids, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        js = {
            "query_id": query_id,
            "candidates": [{"restaurant_id": r, "score": float(s)} for r, s in ranked],
        }
        f.write(json.dumps(js, ensure_ascii=False) + "\n")
