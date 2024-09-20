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

metrics = [
    faithfulness,
    answer_relevancy
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

print("Loaded model and embedding model")

def load_data(type="regular",path="eval_data/failure_outputs.json",batch_size=5): # type = "regular" or "cad" or ... (depends on which decoding method we want to evaluate)
    with open(path,'r') as file:
        data = json.load(file)[:batch_size]
    
    dic = {'question':[],'ground_truth':[],'answer':[],'contexts':[]}

    for i in range(len(data)):
        point = data[i]
        dic['question'].append(point['question'])
        dic['ground_truth'].append(point['sub_answer'][0])
        dic['answer'].append(point['outputs'][type+"_output"])
        dic['contexts'].append([point['sub_context']])
    
    dataset = Dataset.from_dict(dic)
    
    print("Loaded data")

    return dataset

def run_pipeline():
    """
        Store your output jsons in the eval_outputs folder
        Usage: python evaluator.py <json_filename> <type>
    """
    args = sys.argv[1:]
    outputs_filename = args[0]
    type = args[1]

    dataset = load_data(type,path="runs/"+outputs_filename,batch_size=8) # You can't exceed 8 examples!

    result = evaluate(
        dataset, metrics=metrics, llm=azure_model, embeddings=azure_embeddings
    )

    print(result)

if __name__=="__main__":
    run_pipeline()