import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from utils import concatenate_restaurant_text

restaurants_df = pd.read_parquet("data/restaurants.parquet").set_index("restaurant_id")
queries_train_df = pd.read_csv("data/queries_train.csv").set_index("query_id")
qrels_train_df = pd.read_csv(
    "data/qrels_train.tsv", sep="\t", names=["query_id", "restaurant_id", "relevance"]
)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
SEED = 21

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=384, device=device)

# Set seed
torch.manual_seed(SEED)


# Prepare training data
train_examples = []
for _, row in qrels_train_df.iterrows():
    query_id = row["query_id"]
    query = queries_train_df.loc[query_id, "query_text"]

    # Concatenate neighborhood, cuisines, description, and tags
    restaurant_id = row["restaurant_id"]
    r_text = concatenate_restaurant_text(restaurants_df.loc[restaurant_id])
    label = float(row["relevance"])
    train_examples.append(InputExample(texts=[query, r_text], label=label))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

model.fit(
    train_dataloader,
    epochs=5,
    evaluation_steps=2000,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=f"models/{MODEL_NAME}",
    use_amp=True,
)

# Save the model
model.save(f"models/{MODEL_NAME}")
print("Model saved successfully!")
