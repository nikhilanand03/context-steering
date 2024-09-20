import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output
from datasets import load_dataset
from tqdm import tqdm

class ContextDecoder:
    def __init__(self,model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = torch.device("cuda")
        model.to(self.device)
        self.model,self.tokenizer = model,tokenizer
    
    def regular_decoding(self,prompt,debug=True,max_tokens=1,show_tqdm=True): # ID is the index within the inputs list
        if debug: print("prompt: ",prompt)
        self.tokenizer.pad_token = "<s>"
        eos_token = self.tokenizer.eos_token_id
        input_ids = self.tokenizer(prompt,return_tensors="pt",padding=True).input_ids.to(self.device)
        predicted_tokens = []

        token_iterator = tqdm(range(max_tokens)) if (max_tokens>1 and show_tqdm) else range(max_tokens)
        
        for token in token_iterator:
            last_token_logits = self.model(input_ids).logits[0,-1,:]
            last_token_probs = F.softmax(last_token_logits)

            max_index = torch.argmax(last_token_probs).item() # greedy decoding
            
            if max_index == eos_token:
                break

            predicted_tokens.append(max_index)
            
            input_ids = torch.cat([input_ids,torch.tensor([[max_index]]).to(self.device)],dim=1)
        
        out = self.tokenizer.decode(predicted_tokens) if max_tokens>1 \
                    else self.tokenizer.batch_decode(predicted_tokens)[0]
        if debug: print("output: ",out)
        if not debug: return out
    
    def context_aware_decoding(self,context_prompt,plain_prompt,debug=True,
                               alpha=0.5,max_tokens=1,show_tqdm=True):
        # if debug: print("prompt wo context: ", input_wo_context)
        if debug: print("prompt: ",context_prompt)
        self.tokenizer.pad_token = "<s>"
        eos_token = self.tokenizer.eos_token_id
        device=self.device

        predicted_tokens = []
        input_ids_c = self.tokenizer(context_prompt,return_tensors="pt",padding=True).input_ids.to(device)
        input_ids_nc = self.tokenizer(plain_prompt,return_tensors="pt",padding=True).input_ids.to(device)

        token_iterator = tqdm(range(max_tokens)) if (max_tokens>1 and show_tqdm) else range(max_tokens)

        for token in token_iterator:
            # WITH CONTEXT
            context_logits = self.model(input_ids_c).logits[0,-1,:]
        
            # WITHOUT CONTEXT
            plain_logits = self.model(input_ids_nc).logits[0,-1,:]
        
            net_logits = (1+alpha)*context_logits - alpha*plain_logits
            net_prob = F.softmax(net_logits)

            max_index = torch.argmax(net_prob).item() # greedy decoding

            if max_index == eos_token:
                break

            predicted_tokens.append(max_index)
            input_ids_c = torch.cat([input_ids_c,torch.tensor([[max_index]]).to(device)],dim=1)
            input_ids_nc = torch.cat([input_ids_nc,torch.tensor([[max_index]]).to(device)],dim=1)
        
        out = self.tokenizer.decode(predicted_tokens) if max_tokens>1 \
                    else self.tokenizer.batch_decode(predicted_tokens)[0]
        if debug: print("output: ",out)
        if not debug: return out