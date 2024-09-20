import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output
# from datasets import load_dataset
from tqdm import tqdm
import os
import json
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# lemmatizer = WordNetLemmatizer()

def load_dataset():
    with open("../data/dataset.json",'r') as file:
        data = json.load(file)
    return data

def get_accuracy(regular_outs,method_outs,true_outs):
    regular_score,method_score = 0,0
    working_ids = []
    n = len(regular_outs)
    
    for i in range(n):
        reg = False
        if regular_outs[i]==true_outs[i]:
            regular_score+=1
            reg = True
        if method_outs[i]==true_outs[i]:
            method_score+=1
            if not reg:
                working_ids.append(i)
        
    print(f"Regular Decoding Correct: {100*regular_score/n}%")
    print(f"Special Decoding Correct: {100*method_score/n}%")
    return working_ids

def regular_decoding(model,tokenizer,prompt,debug=True,
                     max_tokens=1,show_tqdm=True, return_prob=False): # ID is the index within the inputs list
    if debug: print("prompt: ",prompt)
    device = torch.device("cuda")
    tokenizer.pad_token = "<s>"
    eos_token = tokenizer.eos_token_id
    input_ids = tokenizer(prompt,return_tensors="pt",padding=True).input_ids.to(device)
    predicted_tokens = []

    token_iterator = tqdm(range(max_tokens)) if (max_tokens>1 and show_tqdm) else range(max_tokens)
    
    for token in token_iterator:
        last_token_logits = model(input_ids).logits[0,-1,:]
        last_token_probs = F.softmax(last_token_logits)

        max_index = torch.argmax(last_token_probs).item() # greedy decoding
        
        if max_index == eos_token:
            break

        predicted_tokens.append(max_index)
        
        input_ids = torch.cat([input_ids,torch.tensor([[max_index]]).to(device)],dim=1)
    
    out = tokenizer.decode(predicted_tokens) if max_tokens>1 \
                else tokenizer.batch_decode(predicted_tokens)[0]
    if debug: print("output: ",out)
    if not debug:
        if return_prob==True and max_tokens==1:
            return last_token_probs
        else:
            return out

def context_aware_decoding(model,tokenizer,context_prompt,plain_prompt,debug=True,
                               alpha=0.5,max_tokens=1,show_tqdm=True):
    # if debug: print("prompt wo context: ", input_wo_context)
    if debug: print("prompt: ",context_prompt)
    tokenizer.pad_token = "<s>"
    eos_token = tokenizer.eos_token_id
    device=torch.device("cuda")

    predicted_tokens = []
    input_ids_c = tokenizer(context_prompt,return_tensors="pt",padding=True).input_ids.to(device)
    input_ids_nc = tokenizer(plain_prompt,return_tensors="pt",padding=True).input_ids.to(device)

    token_iterator = tqdm(range(max_tokens)) if (max_tokens>1 and show_tqdm) else range(max_tokens)

    for token in token_iterator:
        # WITH CONTEXT
        with torch.no_grad():
            context_logits = model(input_ids_c).logits[0,-1,:]
        
            # WITHOUT CONTEXT
            plain_logits = model(input_ids_nc).logits[0,-1,:]
    
        net_logits = (1+alpha)*context_logits - alpha*plain_logits
        net_prob = F.softmax(net_logits)

        max_index = torch.argmax(net_prob).item() # greedy decoding

        if max_index == eos_token:
            break

        predicted_tokens.append(max_index)
        input_ids_c = torch.cat([input_ids_c,torch.tensor([[max_index]]).to(device)],dim=1)
        input_ids_nc = torch.cat([input_ids_nc,torch.tensor([[max_index]]).to(device)],dim=1)
    
    out = tokenizer.decode(predicted_tokens) if max_tokens>1 \
                else tokenizer.batch_decode(predicted_tokens)[0]
    if debug: print("output: ",out)
    if not debug: return out