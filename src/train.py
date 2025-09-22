import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

restaurants_df = pd.read_parquet("data/restaurants.parquet").set_index("restaurant_id")


queries_train_df = pd.read_csv("data/queries_train.csv")
qrels_train_df = pd.read_csv(
    "data/qrels_train.tsv", sep="\t", names=["query_id", "restaurant_id", "relevance"]
)
qrels_dev_df = pd.read_csv(
    "data/qrels_dev.tsv", sep="\t", names=["query_id", "restaurant_id", "relevance"]
)
model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"


model = CrossEncoder(model_name, num_labels=1, max_length=512, device="cuda")

train_examples = []
for _, row in qrels_train_df.iterrows():
    query = queries_train_df.loc[
        queries_train_df["query_id"] == row["query_id"], "query_text"
    ].values[0]
    restaurant = restaurants_df.loc[row["restaurant_id"], "description"]
    train_examples.append(
        InputExample(texts=[query, restaurant], label=row["relevance"])
    )

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

model.fit(
    train_dataloader,
    epochs=15,  # Number of training epochs
    optimizer_params={"lr": 2e-5},
    evaluation_steps=1000,  # Evaluate model every 1000 steps
    warmup_steps=1000,  # Gradually increase learning rate over first 1000 steps
    show_progress_bar=True,  # Display training progress
    output_path="models/cross-encoder",  # Directory to save model checkpoints
    use_amp=True,  # Use automatic mixed precision training
)

# Explicitly save the model
model.save("models/cross-encoder")
print("Model saved successfully!")
