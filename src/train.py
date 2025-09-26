import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

restaurants_df = pd.read_parquet("data/restaurants.parquet").set_index("restaurant_id")
queries_train_df = pd.read_csv("data/queries_train.csv").set_index("query_id")
qrels_train_df = pd.read_csv(
    "data/qrels_train.tsv", sep="\t", names=["query_id", "restaurant_id", "relevance"]
)

model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"


model = CrossEncoder(model_name, num_labels=1, max_length=384, device="cuda")

train_examples = []
for _, row in qrels_train_df.iterrows():
    query_id = row["query_id"]
    query = queries_train_df.loc[query_id, "query_text"]

    # Concatenate neighborhood, cuisines, description, and tags
    restaurant_id = row["restaurant_id"]
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
    label = row["relevance"]
    train_examples.append(InputExample(texts=[query, r_text], label=label))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

model.fit(
    train_dataloader,
    epochs=8,  # Number of training epochs
    # optimizer_params={"lr": 2e-5},
    evaluation_steps=2000,  # Evaluate model every 1000 steps
    warmup_steps=100,  # Gradually increase learning rate over first 100 steps
    show_progress_bar=True,  # Display training progress
    output_path="models/cross-encoder",  # Directory to save model checkpoints
    use_amp=True,  # Use automatic mixed precision training
)

# Save the model
model.save("models/cross-encoder")
print("Model saved successfully!")
