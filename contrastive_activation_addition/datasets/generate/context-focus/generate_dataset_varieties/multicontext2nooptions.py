import json
import random

system_prompts = [
    "You are a context-aware assistant and must answer questions strictly based on the provided context.",
    "You are a QA assistant focused on using only the provided context to answer the question.",
    "You are tasked with answering questions solely by referring to the given context.",
    "You are a context-based assistant, required to answer strictly according to the provided context.",
    "You must adhere strictly to the provided context when answering the following question.",
    "Your role is to provide answers derived exclusively from the given context.",
    "You are an assistant designed to answer questions strictly based on the context provided.",
    "You should answer the question strictly in alignment with the given context.",
    "As a QA assistant, you are instructed to refer only to the provided context when answering.",
    "Your task is to answer based only on the information available in the context.",
    "You are a QA assistant and must restrict your answers to the given context.",
    "Provide answers based solely on the context you are given.",
    "You are a context-bound assistant and may only use the provided information to answer.",
    "Answer the question strictly by relying on the given context.",
    "You must base your answers strictly on the provided context, without external information.",
    "Your task is to give context-specific answers and avoid any unrelated information.",
    "You are a context-driven assistant, tasked with answering within the limits of the provided context.",
    "Respond to the following question exclusively using the context provided.",
    "You must strictly rely on the given context to answer the question.",
    "Your answers must adhere strictly to the provided context, without assumptions."
]

with open("generate_dataset_multicontext_2500.json",'r') as f:
    data = json.load(f)

new_list = []
for item in data:
    new_list.append(
        {
            "system_prompt": random.choice(system_prompts),
            "rag": item['rag'],
            "query": item['question']
        }
    )

with open("generate_dataset_no_options.json",'w') as f:
    json.dump(new_list,f)