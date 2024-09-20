import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output
from datasets import load_dataset
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import copy

MODEL = "gpt2xl" # else set it to "mistral7b"

class SteeringAttention(nn.Module):
    def __init__(self,hidden_size,orig_self_attn,alpha,steer_vec):
        super().__init__()
        self.device = torch.device('cuda')
        self.alpha = torch.tensor(alpha,dtype=torch.bfloat16).to(self.device)
        self.orig_self_attn = orig_self_attn
        self.steer_vec= steer_vec
        self.hidden_size=hidden_size
    def forward(self,*args,**kwargs): # kwargs are keyword arguments
        if MODEL=="mistral7b": input_tensor = kwargs.get('hidden_states') # For Mistral
        if MODEL=="gpt2xl": input_tensor = args[0] # For GPT2-XL
        device=self.device
        
        steering_vector = torch.zeros((1,input_tensor.shape[1],self.hidden_size),
                                      dtype=torch.bfloat16).to(device)
        steering_vector[:,-self.steer_vec.shape[0]:,:] += torch.tensor(self.steer_vec,dtype=
                                                            torch.bfloat16).unsqueeze(0).to(device)
        
        orig_output_tup = self.orig_self_attn(*args,**kwargs)
        steer = self.alpha*steering_vector

        if MODEL=="gpt2xl":return (orig_output_tup[0] + steer.type(orig_output_tup[0].dtype),orig_output_tup[1]) # GPT2-XL
        if MODEL=="mistral7b": return (orig_output_tup[0] + steer.type(orig_output_tup[0].dtype),orig_output_tup[1],orig_output_tup[2])
        # return self.orig_self_attn(**kwargs)

class Steerer:
    def __init__(self,p_plus,p_minus,dirn="left",model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        device = torch.device("cuda")
        model.to(device)
        self.device = device
        self.new_model = copy.deepcopy(model)
        self.attn_activations = {}
        self.model,self.tokenizer = model,tokenizer
        self.p_plus,self.p_minus = p_plus,p_minus
        self.layers= [20]
        self.dirn = dirn

    def __get_hook(self,layer_num):
        def hook(model,input,output):
            self.attn_activations[layer_num] = output[0].detach() # not just last token, entire set of activations
        return hook

    def register_hook(self):
        for layer in self.layers:
            if MODEL=="mistral7b": self.model.model.layers[layer].self_attn.register_forward_hook(self.__get_hook(layer))
            if MODEL=="gpt2xl": self.model.transformer.h[layer].attn.register_forward_hook(self.__get_hook(layer))
            print("Layer",layer,"hook registered: ",self.__get_hook(layer))

    def get_vec(self):
        self.tokenizer.pad_token = "<s>"
        p_plus,p_minus,device = self.p_plus,self.p_minus,self.device
        input_ids = self.tokenizer([p_plus,p_minus],return_tensors="pt",padding=True).input_ids.to(device)
        _ = self.model(input_ids)
        p_vecs = self.attn_activations.copy()[self.layers[0]]
        print(p_vecs.shape)
        steer_vec = p_vecs[0] - p_vecs[1] # (p+) - (p-)
        steer_vec = steer_vec if self.dirn=="left" else -steer_vec

        return steer_vec

    def invert_direction(self):
        self.dirn = "left" if self.dirn=="right" else "right"

    def steer_decoding(self,prompt,max_tokens=100,temperature=1.0,alpha=3.0,debug=True): # ID is the index within the inputs list
        self.tokenizer.pad_token = "<s>"
        eos_token = self.tokenizer.eos_token_id
        device = self.device
        model = self.model
        hidden_size = model.config.hidden_size
        n = len(self.model.transformer.h) if MODEL=="gpt2xl" else len(self.model.layers)
    
        for layer in range(n):
            if layer in self.layers:
                if MODEL=="mistral7b":
                    self.new_model.model.layers[layer].self_attn = SteeringAttention(hidden_size,model.model.layers[layer].self_attn,alpha,self.get_vec())
                if MODEL=="gpt2xl":
                    self.new_model.transformer.h[layer].attn = SteeringAttention(hidden_size,model.transformer.h[layer].attn,alpha,self.get_vec())
            else:
                if MODEL=="mistral7b": self.new_model.model.layers[layer].self_attn = model.model.layers[layer].self_attn
                if MODEL=="gpt2xl": self.new_model.transformer.h[layer].attn = model.transformer.h[layer].attn
    
        if debug: print("Prompt:",prompt)
        predicted_tokens = []
        input_ids = self.tokenizer(prompt,return_tensors="pt",padding=True).input_ids.to(device)
    
        token_iterator = tqdm(range(max_tokens)) if max_tokens>1 else range(max_tokens)
        
        for token in token_iterator:
            last_token_logits = self.new_model(input_ids).logits[0,-1,:]
            last_token_probs = F.softmax(last_token_logits)
    
            # max_index = sample_from_logits(last_token_logits,temperature=temperature)[0] # sample decoding
            max_index = torch.argmax(last_token_probs).item() # greedy decoding
    
            if max_index == eos_token:
                break
            
            predicted_tokens.append(max_index)
            input_ids = torch.cat([input_ids,torch.tensor([[max_index]]).to(self.device)],dim=1)
    
        if debug: print(self.tokenizer.decode(predicted_tokens))
        else: return self.tokenizer.decode(predicted_tokens)
        

    
    