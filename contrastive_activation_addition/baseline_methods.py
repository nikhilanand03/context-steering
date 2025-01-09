import json
import os
from common_methods import *
import torch.nn.functional as F
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from itertools import product
import json
import math
from datasets import load_dataset
from utils.tokenize import tokenize_llama_chat,EOT_ID,EOT_ID_GEMMA
import argparse
from utils.helpers import MISTRAL_LIKE_MODEL
from transformers import BitsAndBytesConfig

def get_data_path(type,long,override_ds=None):
    if type=="open_ended":
        if override_ds is not None:
            return f"datasets/test/context-focus/test_dataset_varieties/{override_ds}"
        if long:
            # return "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_1200.json"
            return "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json"
        return "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_failures_fixed_geval_87%.json"
    elif type=="if_eval":
        return "datasets/test/if-eval/if_eval_prompts.json"
    
def load_data(path=get_data_path("open_ended",False)):
    with open(path,'r') as file:
        data = json.load(file)
    return data

def tokenize_func(tokenizer,user_input,system_prompt=None):
    return tokenize_llama_chat(tokenizer,user_input,model_output=None,system_prompt=system_prompt)

def generate_cad_outputs(model,tokenizer,data,use_mistral,alpha,type="open_ended",long=False,override_ds=None,suffix=""):
    """
    Generates CAD outputs.
    """
    outputs = []

    for i in tqdm(range(len(data))):
        d = data[i]
        # print(d)
        if type=="open_ended":
            context_tokens = torch.tensor([tokenize_func(tokenizer,d['question'])]).to(model.device)
            plain_tokens = torch.tensor([tokenize_func(tokenizer,d['question'].split(">\n")[-1])]).to(model.device)
            toks = 65
        elif type=="if_eval":
            context_tokens = torch.tensor([tokenize_func(tokenizer,d['question'])]).to(model.device)
            plain_tokens = torch.tensor([tokenize_func(tokenizer,"")]).to(model.device)
            toks = 400
        out = context_aware_decoding(model,tokenizer,context_tokens,plain_tokens,alpha=alpha,max_tokens=toks,show_tqdm=False)
        d['model_output'] = out
        try:
            del d['raw_model_output']
        except:
            pass
        outputs.append(d)
    
    os.makedirs(f"results{suffix}/context-focus", exist_ok=True)

    with open(f"results{suffix}/context-focus/cad_type={type}_alpha={alpha}_long={long}_usemistral={use_mistral}_override_ds={override_ds}.json",'w') as file:
        json.dump(outputs,file)

def generate_neg_outputs(model,tokenizer,data,use_mistral,alpha,long=False):
    """
    Generates negative decoding outputs on the current outputs file and adds it to the same file.
    """
    outputs = []

    for i in tqdm(range(len(data))):
        d = data[i]

        plus = "You are an AI system who is instructed to only answer according to the statements sent to you. Refrain from answering without proper justification from the following few sentences enclosed within the tags. "
        minus = "You are an AI system who can answer the question as per your knowledge. Consider the following few statements (enclosed within the tags) but feel free to answer however you wish. "

        plus_tokens = torch.tensor([tokenize_func(tokenizer,d['question'],system_prompt=plus)]).to(model.device)
        minus_tokens = torch.tensor([tokenize_func(tokenizer,d['question'],system_prompt=minus)]).to(model.device)
        out = context_aware_decoding(model,tokenizer,plus_tokens,minus_tokens,alpha=alpha,max_tokens=65,show_tqdm=False)
        d['model_output'] = out
        try:
            del d['raw_model_output']
        except:
            pass
        outputs.append(d)

    with open(f"results/context-focus/neg_open_ended_alpha={alpha}_long={long}_usemistral={use_mistral}.json",'w') as file:
        json.dump(outputs,file)

def generate_fneg_outputs(model,tokenizer,data,use_mistral,alpha,t,long=False):
    """
    Generates filtered-negative-decoding outputs on the current outputs file and adds it to the same file.
    """
    outputs = []

    for i in tqdm(range(len(data))):
        d = data[i]

        plus = "You are an AI system who is instructed to only answer according to the statements sent to you. Refrain from answering without proper justification from the following few sentences enclosed within the tags. "
        minus = "You are an AI system who can answer the question as per your knowledge. Consider the following few statements (enclosed within the tags) but feel free to answer however you wish. "

        plus_tokens = torch.tensor([tokenize_func(tokenizer,d['question'],system_prompt=plus)]).to(model.device)
        minus_tokens = torch.tensor([tokenize_func(tokenizer,d['question'],system_prompt=minus)]).to(model.device)
        out = context_aware_decoding(model,tokenizer,plus_tokens,minus_tokens,alpha=alpha,max_tokens=65,show_tqdm=False,t=t,bars_name = f"fneg_{i}_{alpha}_{t}")
        d['model_output'] = out
        try:
            del d['raw_model_output']
        except:
            pass
        outputs.append(d)
    
    os.makedirs(f"results{suffix}/context-focus", exist_ok=True)

    with open(f"results{suffix}/context-focus/fneg_open_ended_alpha={alpha}_long={long}_usemistral={use_mistral}.json",'w') as file:
        json.dump(outputs,file)
    
def get_model_path(use_mistral):
    if use_mistral:
        return MISTRAL_LIKE_MODEL
    else:
        return "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["open_ended", "if_eval"], default="open_ended")
    parser.add_argument("--long", action="store_true", default=False)
    parser.add_argument("--use_mistral", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--override_dataset_path", type=str, default=None)
    args = parser.parse_args()

    MODEL_ID = get_model_path(args.use_mistral)

    if MODEL_ID=="meta-llama/Meta-Llama-3.1-70B-Instruct":
        # model_config = {
        #     "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        #     "device_map": "auto",
        #     "torch_dtype": torch.bfloat16
        # }
        model_config = {"device_map": "auto"} ## If we dont wanna quantise
    else:
        model_config = {}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if MODEL_ID in ["meta-llama/Meta-Llama-3.1-8B-Instruct","meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        tokenizer.eos_token = EOT_ID
    elif MODEL_ID=="google/gemma-2-2b-it":
        tokenizer.eos_token = EOT_ID_GEMMA
    elif MODEL_ID=="mistralai/Mistral-7B-Instruct-v0.3":
        tokenizer.eos_token = "</s>"
    else:
        tokenizer.eos_token = "</s>"
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,**model_config).eval()
    device = torch.device("cuda")
    model.to(device)

    print("Loaded model")

    CAD = False
    NEG = False
    FNEG = False
    CAD_WO_BASELINE = True

    if CAD_WO_BASELINE:
        data = load_data(get_data_path(args.type,args.long,args.override_dataset_path))
        generate_cad_outputs(model,tokenizer,data,args.use_mistral,alpha=1,type=args.type,long=args.long,override_ds=args.override_dataset_path,suffix=args.suffix)
    if CAD:
        data = load_data(get_data_path(args.type,args.long))
        generate_cad_outputs(model,tokenizer,data,args.use_mistral,alpha=1,type=args.type,long=args.long) # Any larger will cause grammatical issues, so this is the max we can go
        generate_cad_outputs(model,tokenizer,data,args.use_mistral,alpha=0,type=args.type,long=args.long) # Baseline for comparison
    if NEG:
        data = load_data(get_data_path(args.type,args.long))
        generate_neg_outputs(model,tokenizer,data,args.use_mistral,alpha=1,long=args.long)
    if FNEG:
        data = load_data(get_data_path(args.type,args.long))
        generate_fneg_outputs(model,tokenizer,data,args.use_mistral,alpha=6,t=1e-2,long=args.long)

if __name__=="__main__":
    run_pipeline()