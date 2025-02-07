#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import argparse

seed = 2023

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('Cuda:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
print('pwd', os.getcwd())

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
# from util_clm import convert_model_to_int8_on_gpu

import jsonlines

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


# completion_template = "Q: {} A:"  # "{}" # "Query: {}\nResult:" # "Q: {} A:" # "{} The answer is"
completion_template_instruct = "Question: {question}\nAnswer: {answer}"
completion_template = "Answer the following question:\n\n{question}"  # "{}" # "Query: {}\nResult:" # "Q: {} A:" # "{} The answer is"
completion_template_context_instruct = "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
completion_template_context = "Answer based on context:\n\n{context}\n\n{question}"
genread_template = "Generate a background document from Wikipedia to answer the given question. {}"  # This prompt comes from the GenRead paper

B_INST, E_INST = "[INST]", "[/INST]"

B_TEXT = "<|begin_of_text|>"
B_HEADER, E_HEADER = "<|start_header_id|>","<|end_header_id|>"
EOT_ID = "<|eot_id|>"

PAD_TOKEN_LATEST = "<|finetune_right_pad_id|>"
PAD_TOKEN_ID_LATEST = 128004

def template_mistral(user_input: str):
    return f"{B_INST} {user_input.strip()} {E_INST}"

def template_llama_3_1_8B(user_input: str):      
    return f"{B_HEADER}user{E_HEADER}\n\n{user_input.strip()}{EOT_ID}\n{B_HEADER}assistant{E_HEADER}\n\n"

def wrap_input(user_input: str, model_name: str, is_instruct: bool):
    if is_instruct:
        if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
            return template_mistral(user_input)
        elif model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            return template_llama_3_1_8B(user_input)
    else:
        return user_input

def get_few_shot_text(row, is_instruct=False):
    if not is_instruct:
        return completion_template.format(question=row.question) + "\n\n" + str(row.ans)
    else:
        return completion_template_instruct.format(question=row.question,answer=row.ans)

def get_few_shot_text_with_retrieval(row, retrieval_dict,is_instruct=False):

    if not is_instruct:
        few_shot_text_context = lambda retrieved_text: completion_template_context.format(context=retrieved_text, question=row.question) + "\n\n" + str(row.ans)
    else:
        few_shot_text_context = lambda retrieved_text: completion_template_context_instruct.format(context=retrieved_text, question=row.question, answer=row.ans)

    if row.question in retrieval_dict:
        retrieval = {"text": retrieval_dict[row.question]["gold_ctx"]}
        retrieved_text = retrieval["text"]
        # return retrieved_text + "\n\n" + completion_template.format(row.question) + " " + str(row.ans)
        return few_shot_text_context(retrieved_text)
    elif row.question.replace("?", "").lower() in retrieval_dict:
        retrieval = {"text": retrieval_dict[row.question.replace("?", "").lower()]["gold_ctx"]}
        retrieved_text = retrieval["text"]
        # return retrieved_text + "\n\n" + completion_template.format(row.question) + " " + str(row.ans)
        return few_shot_text_context(retrieved_text)
    else:
        print("missing retrieval")
        return get_few_shot_text(row, is_instruct)

def get_full_few_shot_prompt(item, test_data, is_instruct):
    knowledge_df = pd.DataFrame(test_data)
    
    examples = knowledge_df[knowledge_df["question"] != item["question"]].sample(n=5)

    retrieval_dict = {entry["question"]: entry for entry in test_data}

    few_shot_examples = [
        get_few_shot_text_with_retrieval(row2, retrieval_dict, is_instruct)
        for _, row2 in examples.iterrows()
    ]

    np.random.shuffle(few_shot_examples)

    if not is_instruct:
        few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
    else:
        intro_instruct = "Use the following examples to answer the question given at the end:"
        few_shot_examples_text = intro_instruct + "\n" + "\n\n".join(
            f"Example {i + 1}:\n{example}" for i, example in enumerate(few_shot_examples)
        ) + "\n\n"

    return few_shot_examples_text + completion_template_context.format(context=item['gold_ctx'],question=item["question"])  