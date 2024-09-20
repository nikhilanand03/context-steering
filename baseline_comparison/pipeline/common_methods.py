import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output
# from datasets import load_dataset
from tqdm import tqdm
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

def entropy(probs):
    return -torch.sum(probs * torch.log(probs), dim=-1)

def load_memotrap(): # returns (context_inputs,plain_inputs,context_outputs)
    with open("../data/memotrap_dataset.txt",'r') as file:
        dataset_string = file.readlines()

    arr = [item.split("\t")[:2] for item in dataset_string]
    context_inputs = [item[0] for item in arr]
    outputs = [[s.strip(".").strip(" ") for s in eval(item[1])] for item in arr]
    # print(outputs)

    assert len(context_inputs)==215

    context_outputs = []
    for i in range(len(context_inputs)):
        inp = context_inputs[i]
        first_quote = inp.find('"')
        second_quote = inp[first_quote+1:].find('"') + first_quote + 1
        context_output = inp[first_quote+1:second_quote]
        context_outputs.append(context_output)
        outputs[i] = outputs[i][0] if outputs[i][1]==context_outputs[i] else outputs[i][1]
        
    assert len(context_outputs)==215

    plain_inputs = []
    for i in range(len(context_inputs)):
        plain_input = context_inputs[i][:13]+context_inputs[i][context_inputs[i].find(":"):]
        plain_inputs.append(plain_input)

    assert len(plain_inputs)==215

    return context_inputs,plain_inputs,context_outputs,outputs

def load_technews():
    ds_path = "ds_tech_news"
    dataset = {}
    
    for filename in os.listdir(ds_path):
        file_path = os.path.join(ds_path, filename)
        
        if filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                contents = file.read()
                cqa = contents.split("</context>")
                context = cqa[0][9:]
                qa = cqa[1].split("</a>")[:-1]
                qa_new = [item.split("<a>") for item in qa]
                qa_final = [[item[0].strip("\n").lstrip("<q>").rstrip("</q>"),item[1]] for item in qa_new]
                # print(qa_final)
                # print("***")
                dataset[context]=qa_final
                
    return dataset

def remove_words(ans,ques): # ques's words are removed from ans
    filtered_sentence = []

    words_to_remove = set()
    for word in nltk.word_tokenize(ques):
        lemma = lemmatizer.lemmatize(word.lower(),pos="v")
        words_to_remove.add(lemma)

    stop_words = set(stopwords.words('english'))
    for word in nltk.word_tokenize(ans):
        lemma = lemmatizer.lemmatize(word.lower(),pos="v")
        word = word.rstrip('.,!?;:')
        if lemma not in words_to_remove:
            filtered_sentence.append(word)

    return ' '.join(filtered_sentence)

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

def filter_logits(logits, probs, threshold):
    if threshold==0:
        return logits
    
    mask = probs < threshold
    neg_inf = torch.full_like(logits, float('-inf'))
    filtered_logits = torch.where(mask, neg_inf, logits)

    return filtered_logits

def print_bars(filtered_logits, plain_probs, context_probs, net_probs, tokenizer,name):
    with open(f"runs/bars_{name}.txt",'a') as file:

        valid_indices = torch.where(filtered_logits != float('-inf'))[0]
        valid_tokens = [tokenizer.decode([i]) for i in valid_indices]
        valid_plain_probs = plain_probs[valid_indices]
        valid_net_probs = net_probs[valid_indices]
        valid_context_probs = context_probs[valid_indices]
        
        file.write("Tokens above threshold:\n")
        max_prob = max(valid_plain_probs.max().item(), valid_net_probs.max().item(),valid_context_probs.max().item())
        
        for token, plain_prob, net_prob, context_prob in zip(valid_tokens, valid_plain_probs, valid_net_probs, valid_context_probs):
            plain_bar_length = int((plain_prob / max_prob) * 100)
            net_bar_length = int((net_prob / max_prob) * 100)
            context_bar_length = int((context_prob / max_prob) * 100)
            
            plain_bar = '█' * plain_bar_length + '░' * (100 - plain_bar_length)
            net_bar = '█' * net_bar_length + '░' * (100 - net_bar_length)
            context_bar = '█' * context_bar_length + '░' * (100 - context_bar_length)
            
            file.write(f"  {token:<20} |{plain_bar}| Plain: {plain_prob:.4f}\n")
            file.write(f"  {' ':<20} |{net_bar}| Net: {net_prob:.4f}\n")
            file.write(f"  {' ':<20} |{context_bar}| Context: {context_prob:.4f}\n")
            file.write("\n")  # Empty line for separation

def context_aware_decoding(model,tokenizer,context_prompt,plain_prompt,debug=True,
                               alpha=0.5,max_tokens=1,show_tqdm=True,t=None,bars_name=""):
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
        with torch.no_grad():
            context_logits = model(input_ids_c).logits[0,-1,:]
            context_probs = F.softmax(context_logits)
            plain_logits = model(input_ids_nc).logits[0,-1,:]
            plain_probs = F.softmax(plain_logits)
    
        net_logits = (1+alpha)*context_logits - alpha*plain_logits
        filtered_net_logits = filter_logits(net_logits,plain_probs,t*torch.max(context_logits) if t else 0.0)
        net_probs = F.softmax(filtered_net_logits)

        print_bars(filtered_net_logits, plain_probs, context_probs, net_probs, tokenizer,name=bars_name)

        max_index = torch.argmax(net_probs).item() # greedy decoding

        if max_index == eos_token:
            break

        predicted_tokens.append(max_index)
        input_ids_c = torch.cat([input_ids_c,torch.tensor([[max_index]]).to(device)],dim=1)
        input_ids_nc = torch.cat([input_ids_nc,torch.tensor([[max_index]]).to(device)],dim=1)
    
    out = tokenizer.decode(predicted_tokens) if max_tokens>1 \
                else tokenizer.batch_decode(predicted_tokens)[0]
    if debug: print("output: ",out)
    if not debug: return out

def dynamicA_irrelevant_decoding(model,tokenizer,rel_prompt,irrel_prompt,plain_prompt,debug=True,
                               beta=2,max_tokens=1,show_tqdm=True):
    # if debug: print("prompt wo context: ", input_wo_context)
    if debug: print("prompt: ",rel_prompt)
    tokenizer.pad_token = "<s>"
    eos_token = tokenizer.eos_token_id
    device=torch.device("cuda")

    predicted_tokens = []
    input_ids_c = tokenizer(rel_prompt,return_tensors="pt",padding=True).input_ids.to(device)
    input_ids = tokenizer(plain_prompt,return_tensors="pt",padding=True).input_ids.to(device)
    input_ids_nc = tokenizer(irrel_prompt,return_tensors="pt",padding=True).input_ids.to(device)

    token_iterator = tqdm(range(max_tokens)) if (max_tokens>1 and show_tqdm) else range(max_tokens)

    for token in token_iterator:
        with torch.no_grad():
            rel_logits = model(input_ids_c).logits[0,-1,:]
            plain_logits = model(input_ids).logits[0,-1,:]
            irrel_logits = model(input_ids_nc).logits[0,-1,:]

            rel_probs = F.softmax(rel_logits)
            plain_probs = F.softmax(plain_logits)
            irrel_probs = F.softmax(irrel_logits)
        
        # confusion_context = entropy(rel_probs)
        # confusion_plain = entropy(plain_probs)
        C = torch.max(plain_probs)
        CR = torch.max(rel_probs)

        # If adding the context makes it more confused, memory probably deviates from context and we need to steer more
        # It would be less confused with context if the LLM doesn't know much about the topic so memory can't interfere.
        # It would be equally confused if the LLM's memory agrees with the context.
        # In either case, the LLM will naturally go to context and we don't need to interfere as much here.
        
        # alpha = beta * ( C if (confusion_context > confusion_plain) else (1 - CR) )
        alpha = beta * ( (1 - C) if (C > CR) else CR )
    
        net_logits = plain_logits + alpha*(rel_logits - irrel_logits)
        net_prob = F.softmax(net_logits)

        max_index = torch.argmax(net_prob).item() # greedy decoding

        if max_index == eos_token:
            break

        predicted_tokens.append(max_index)
        input_ids_c = torch.cat([input_ids_c,torch.tensor([[max_index]]).to(device)],dim=1)
        input_ids = torch.cat([input_ids,torch.tensor([[max_index]]).to(device)],dim=1)
        input_ids_nc = torch.cat([input_ids_nc,torch.tensor([[max_index]]).to(device)],dim=1)
    
    out = tokenizer.decode(predicted_tokens) if max_tokens>1 \
                else tokenizer.batch_decode(predicted_tokens)[0]
    if debug: print("output: ",out)
    if not debug: return out