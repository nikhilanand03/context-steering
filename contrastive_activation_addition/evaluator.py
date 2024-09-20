import json
import os
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness
)
from datasets import load_dataset, Dataset
import sys
import argparse

metrics = [
    faithfulness,
    # answer_relevancy
]

azure_configs = {
    "base_url": "https://docexpresearch.openai.azure.com/",
    "model_deployment": "gpt-35-turbo-0613",
    "model_name": "gpt-35-turbo",
    "embedding_deployment": "text-embedding-3-large",
    "embedding_name": "text-embedding-3-large",
}

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

# print("Loaded model and embedding model")

def load_data(type="regular",path="eval_data/failure_outputs.json"): # type = "regular" or "cad" or ... (depends on which decoding method we want to evaluate)
    with open(path,'r') as file:
        data = json.load(file)
    
    dic = {'question':[],'ground_truth':[],'answer':[],'contexts':[]}

    for i in range(len(data)):
        point = data[i]
        dic['question'].append(point['question'])
        dic['ground_truth'].append("")
        dic['answer'].append(point['output'])
        dic['contexts'].append(point['contexts'])
    
    dataset = Dataset.from_dict(dic)
    
    # print("Loaded data")

    return dataset

def run_pipeline():
    """
        Store your output jsons in the eval_outputs folder
        Usage: python evaluator.py --outputs_path <full_output_path>
    """
    outputs_path = args.outputs_path
    dataset = load_data(type,path=outputs_path)

    result = evaluate(
        dataset, metrics=metrics, llm=azure_model, embeddings=azure_embeddings
    )

    print(result)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs_path",
        type=str,
        required=True
    )
    args = parser.parse_args()
    run_pipeline()