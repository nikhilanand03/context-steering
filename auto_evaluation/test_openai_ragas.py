from datasets import load_dataset, Dataset
import asyncio

from ragas.metrics import (
    answer_relevancy,
    faithfulness
)

# list of metrics we're going to use
metrics = [
    faithfulness,
    answer_relevancy
]

azure_configs = {
    "base_url": "https://docexpresearch.openai.azure.com/",
    "model_deployment": "gpt-4o",
    "model_name": "gpt-4o",
    "embedding_deployment": "text-embedding-3-large",
    "embedding_name": "text-embedding-3-large",
}

import os

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig

amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")

print("Loaded dataset")

azure_model = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
)

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2024-02-01",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)

print("Loaded model and embedding model")

print(len(amnesty_qa['eval']))

subset = Dataset.from_dict(amnesty_qa['eval'][:10])
print(len(subset))

def main():
    result = evaluate(
        subset, metrics=metrics, llm=azure_model, embeddings=azure_embeddings
    )

    print("Evaluated results")

    print(result)

if __name__ == "__main__":
    main()