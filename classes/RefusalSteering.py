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

class SteeringLayer(nn.Module):
    def __init__(self,hidden_size,orig_layer,r_opt,type="addition"): # 'addition' or 'ablation'
        super().__init__()
        self.device = torch.device('cuda')
        self.orig_layer = orig_layer
        self.hidden_size=hidden_size
        self.r_opt = r_opt
        self.type=type
    def forward(self,*args,**kwargs): # kwargs are keyword arguments
        input_tensor = args[0]
        device=self.device
        tokens_dim = input_tensor.shape[1]
        orig_output_tup = self.orig_layer(*args,**kwargs)

        vec = torch.zeros((self.hidden_size))
        if type=="addition":
            vec = self.r_opt
        elif type=="ablation":
            r_cap = self.r_opt/torch.norm(self.r_opt)
            dot = torch.dot(r_cap,orig_output_tup[0]) # scalar
            vec = -dot*r_cap
            
        steer = torch.zeros((1,tokens_dim,self.hidden_size),
                                      dtype=torch.bfloat16).to(device)
        steer[0,-1,:] += vec.to(device)

        return (orig_output_tup[0] + steer.type(orig_output_tup[0].dtype),orig_output_tup[1])
