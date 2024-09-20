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

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = torch.device("cuda")
model.to(device)

print("Loaded model")

def load_data():
    with open("runs/outputs.json",'r') as file:
        data = json.load(file)
    return data

def generate_cad_outputs(data,alpha):
    """
    Generates CAD outputs on the current outputs file and adds it to the same file.
    """
    outputs = []

    for i in range(len(data)):
        dic = {'id':i}
        dic.update(data[i])
        context_prompt = "[INST]"+dic['sub_context']+dic['question']+"[/INST]"
        plain_prompt = "[INST]"+dic['question']+"[/INST]"
        out = context_aware_decoding(model,tokenizer,context_prompt,plain_prompt,debug=False,alpha=alpha,max_tokens=100,show_tqdm=True)
        dic['outputs']['cad_output'+str(alpha)] = out
        outputs.append(dic)

    with open("runs/outputs.json",'w') as file:
        json.dump(outputs,file)

def generate_neg_outputs(data,alpha):
    """
    Generates negative decoding outputs on the current outputs file and adds it to the same file.
    """
    outputs = []

    for i in range(len(data)):
        dic = {}
        dic.update(data[i])

        plus = "You are an AI system who is instructed to only answer according to the statements sent to you. Refrain from answering without proper justification from the following few sentences enclosed within the <P></P> tag. "
        minus = "You are an AI system who can answer the question as per your knowledge. Consider the following few statements (enclosed within the <P></P> tag) but feel free to answer however you wish. "

        plus_prompt = "[INST]"+plus+dic['sub_context']+dic['question']+"[/INST]"
        minus_prompt = "[INST]"+minus+dic['sub_context']+dic['question']+"[/INST]"
        out = context_aware_decoding(model,tokenizer,plus_prompt,minus_prompt,debug=False,alpha=alpha,max_tokens=100,show_tqdm=True)
        dic['outputs']['neg_output'+str(alpha)] = out
        outputs.append(dic)

    with open("runs/outputs.json",'w') as file:
        json.dump(outputs,file)

def generate_irr_outputs(data,beta):
    """
    Generates irrelevant decoding outputs on the current outputs file and adds it to the same file.
    """
    outputs = []

    for i in range(len(data)):
        dic = {}
        dic.update(data[i])
        rel_context = dic['sub_context']
        irr_context = data[(i+3)%len(data)]['sub_context']

        assert not rel_context==irr_context

        plus = "You are an AI system who is instructed to only answer according to the statements sent to you. Refrain from answering without proper justification from the following few sentences enclosed within the <P></P> tag. "
        minus = "You are an AI system who can answer the question as per your knowledge. Consider the following few statements (enclosed within the <P></P> tag) but feel free to answer however you wish. "
        
        plain_prompt = "[INST]"+dic['question']+"[/INST]"
        rel_prompt = "[INST]"+plus+rel_context+dic['question']+"[/INST]"
        irr_prompt = "[INST]"+minus+irr_context+dic['question']+"[/INST]"
        print(irr_prompt)
        out = dynamicA_irrelevant_decoding(model,tokenizer,rel_prompt,irr_prompt,plain_prompt,debug=False,beta=beta,max_tokens=100,show_tqdm=True)
        dic['outputs']['irr_output'+str(beta)] = out
        outputs.append(dic)

    with open("runs/outputs.json",'w') as file:
        json.dump(outputs,file)

def generate_fneg_outputs(data,alpha,t):
    """
    Generates filtered-negative-decoding outputs on the current outputs file and adds it to the same file.
    """
    outputs = []

    for i in range(len(data)):
        dic = {}
        dic.update(data[i])

        plus = "You are an AI system who is instructed to only answer according to the statements sent to you. Refrain from answering without proper justification from the following few sentences enclosed within the <P></P> tag. "
        minus = "You are an AI system who can answer the question as per your knowledge. Consider the following few statements (enclosed within the <P></P> tag) but feel free to answer however you wish. "

        plus_prompt = "[INST]"+plus+dic['sub_context']+dic['question']+"[/INST]"
        minus_prompt = "[INST]"+minus+dic['sub_context']+dic['question']+"[/INST]"
        out = context_aware_decoding(model,tokenizer,plus_prompt,minus_prompt,debug=False,alpha=alpha,max_tokens=100,show_tqdm=True,t=t,bars_name = f"fneg_{i}_{alpha}_{t}")
        dic['outputs']['fneg_output'+str(alpha)+f'_{t:.0e}'] = out
        outputs.append(dic)

    with open("runs/outputs.json",'w') as file:
        json.dump(outputs,file)

def run_pipeline():
    GEN_CAD = False
    GEN_NEG = False # Negative decoding
    GEN_IRR = False
    GEN_FNEG1 = False
    GEN_FNEG2 = False
    GEN_FNEG3 = True

    if GEN_CAD:
        data = load_data() # Load the data in its current state
        for alpha in [0.5,2,4]:
            generate_cad_outputs(data,alpha=alpha)
    
    if GEN_NEG:
        data = load_data()
        for alpha in [0.5,2,4]:
            generate_neg_outputs(data,alpha=alpha)

    if GEN_IRR:
        data = load_data()
        for beta in [1,2,4]:
            generate_irr_outputs(data,beta=beta)
    
    if GEN_FNEG1:
        data = load_data()
        for alpha in [4,6,8]:
            generate_fneg_outputs(data,alpha=alpha,t=1e-3)
        
    if GEN_FNEG2:
        data = load_data()
        for t in [1e-4,1e-5,1e-6]:
            generate_fneg_outputs(data,alpha=8,t=t)
    
    if GEN_FNEG3:
        data = load_data()
        for alpha in [12]:
            generate_fneg_outputs(data,alpha=alpha,t=1e-6)

if __name__=="__main__":
    run_pipeline()