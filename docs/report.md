# TableCheck ML Take-home Report

## Candidate Information
- Name: Jaron Dellota
- Time spent: 4 hrs
- Hardware used: Ryzen 5 3600, RTX 2060 Super (8 GB VRAM), 32 GB RAM

## Problem Understanding
The task involves improving TableCheck's restaurant search functionality by fine-tuning a neural re-ranking model. The business goal is to enhance user experience by providing more relevant restaurant recommendations when customers search with natural language queries.

**Assumptions**

- The BM25 candidate pool has adequate recall; reranking focuses on ordering quality.

- Latency and throughput matter in production; model choice balances quality vs. cost.

## Approach Overview
### Data Preparation

I began with exploratory data analysis and identified that the most informative columns for restaurant representation were `description`, `neighborhood`, `cuisines`, and `tags`. I concatenated these fields to form the input text for each restaurant. Additionally, I observed the presence of Japanese characters in the dataset, which highlighted the importance of considering multilingual support in model selection.

### Training

I initially trained a cross-encoder model using `cross-encoder/ms-marco-MiniLM-L12-v2`, a lightweight and efficient model suitable for retrieval tasks. On the development set, this model outperformed the baselines on most metrics except Recall@50. To address the multilingual aspect and further improve performance, I then trained `BAAI/bge-reranker-v2-m3`, a multilingual reranker, which surpassed the benchmarks across all evaluation metrics.

## Experiments

### How to Run

1. **Set up the environment**

   Prerequisite: [uv](https://github.com/astral-sh/uv) installed.

   ```bash
   uv sync
   ```

2. **Train the model**

   ```bash
   uv run src/train.py
   ```

3. **Run inference**

   ```bash
   uv run src/predict.py --split <dev|test> --model_path <model_name>
   ```

   - Replace `<dev|test>` with the desired data split.
   - Replace `<model_name>` with the name of your trained model directory (e.g., `cross-encoder/ms-marco-MiniLM-L12-v2`).

### Evaluation Results (DEV)
| Model                      | Parameters                                   | NDCG@10 | MRR@10 | Recall@50 |
|----------------------------|----------------------------------------------|---------|--------|-----------|
| ms-marco-MiniLM-L12-v2     | epochs=1, evaluation_steps=2000, warmup_steps=100  | 0.2395  | 0.3550 | 0.2549    |
| ms-marco-MiniLM-L12-v2     | epochs=3, evaluation_steps=2000, warmup_steps=100  | 0.2229  | 0.2096 | 0.4486    |
| ms-marco-MiniLM-L12-v2     | epochs=5, evaluation_steps=2000, warmup_steps=100  | 0.5264  | 0.5930 | 0.5421    |
| bge-reranker-v2-m3         | epochs=1, evaluation_steps=1000, warmup_steps=100  | 0.7441  | 0.8735 | 0.6428    |

Among the evaluated models, `bge-reranker-v2-m3` achieved the best overall performance across all metrics. However, this improvement came with notable tradeoffs: the model required substantially more time and computational resources for both training and inference compared to the lighter `ms-marco-MiniLM-L12-v2`.  The choice between these models involves balancing the need for top-tier ranking quality against operational efficiency and cost constraints.

## Production Considerations

For deployment, I would evaluate business requirements around latency, throughput, and model performance, carefully selecting the model architecture and hardware to balance these tradeoffs. For monitoring and retraining, I would track model performance and latency over time, establishing a retraining schedule as needed to maintain accuracy and ensure a smooth user experience.

## Next Steps

- Conduct more comprehensive EDA and refine input templates (experiment with field ordering, truncation methods, and concise summaries).
- Perform systematic hyperparameter tuning.
- Investigate more efficient or higher-capacity multilingual rerankers; evaluate model distillation and quantization for deployment.


## Appendix (Optional)
References: 
- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2

