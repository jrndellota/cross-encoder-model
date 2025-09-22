from sentence_transformers import CrossEncoder
import pandas as pd
import json

restaurants_df = pd.read_parquet("data/restaurants.parquet").set_index("restaurant_id")
queries_dev_df = pd.read_csv("data/queries_dev.csv")
candidates_dev_df = pd.read_json("data/bm25_candidates_dev_top500.jsonl", lines=True)

model = CrossEncoder("models/cross-encoder")

output_path = "runs/test_predictions.jsonl"

print(f"Number of candidate rows: {len(candidates_dev_df)}")

with open(output_path, "w+", encoding="utf-8") as f:
    from tqdm import tqdm

    for i, row in tqdm(
        candidates_dev_df.iterrows(),
        total=len(candidates_dev_df),
        desc="Processing queries",
    ):
        query_id = row["query_id"]
        candidates = row["candidates"]
        query = queries_dev_df.loc[
            queries_dev_df["query_id"] == query_id, "query_text"
        ].values[0]

        pairs = []
        restaurant_ids = []
        for candidate in candidates:
            restaurant_ids.append(candidate["restaurant_id"])
            description = restaurants_df.loc[candidate["restaurant_id"], "description"]
            pairs.append([query, description])

        scores = model.predict(pairs, convert_to_numpy=True)
        ranked = sorted(
            zip(restaurant_ids, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        js = {
            "query_id": query_id,
            "candidates": [{"restaurant_id": r, "score": float(s)} for r, s in ranked],
        }
        f.write(json.dumps(js, ensure_ascii=False) + "\n")
