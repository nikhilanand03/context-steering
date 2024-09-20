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

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = torch.device("cuda")
model.to(device)
activations = {}

layer_list = [4,8,16,18,20,22,24,26,28,30,31,32]
LAYERS_ENTROPIES = [24,26]
LAYERS_OUTPUTS = [31,32]
GET_ENTROPIES = False
DECODE_INPUTS = True
VOCAB_SIZE = 32000

def format_template(intro,context,query):
    if context and intro:
        prompt = f"<s>[INST]{intro} Context: {context}\nQuestion: {query}[/INST] " # Space at the end
    else:
        prompt = f"<s>[INST]Question: {query}[/INST] "
    print(prompt)
    return prompt

def get_hook(layer_num,type):
    def hook(model,input,output):
        # print(f"Hook called for layer {layer_num}, type {type}.")
        activations[layer_num] = output[0].detach() # not just last token, entire set of activations
    return hook

def register_hooks(t):
    list_of_hooks = []
    for i in layer_list:
        list_of_hooks.append(model.model.layers[i-1].register_forward_hook(get_hook(i,t)))

    return list_of_hooks

def remove_hooks(list_of_hooks):
    print("Removing hooks.")
    for hook in list_of_hooks:
        hook.remove()

def get_entropy_across_intokens(matrix):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    
    device = matrix.device
    epsilon = 1e-10
    p_log_p = matrix * torch.log(matrix + epsilon)

    entropies = -torch.sum(p_log_p, dim=0)
    print(entropies.shape)

    return entropies

def get_entropy_for_context(context,layers_to_check):
    context_token_list = tokenizer.batch_decode(tokenizer(context).input_ids)

    print(context_token_list)
    
    out=regular_decoding(model,tokenizer,context,debug=False,max_tokens=1,show_tqdm=True, return_prob=False)

    # word_tok = tokenizer(word).input_ids[1]
    matrix = torch.zeros((len(context_token_list),len(layers_to_check),VOCAB_SIZE))

    for l,i in product(range(len(layers_to_check)),range(len(context_token_list))):
        layer = layers_to_check[l]
        act = F.softmax(model.lm_head(model.model.norm(activations[layer][0,i,:])),dim=0)
        prob_toks = act
        
        matrix[i,l,:] = prob_toks

    sf_mat = F.softmax(matrix,dim=0).detach()
    print(sf_mat.shape)

    entropies = get_entropy_across_intokens(sf_mat)
    min_ent, _ = torch.min(entropies, dim=0)

    # Great, entropies look accurate! Most entropies are around the same value since most tokens are just uniform everywhere.
    # Only 'fur's entropy is significantly higher due to visible activations.

    return min_ent

def save_all_entropies(data):
    print("Saving entropies...")
    all_entropies = torch.zeros((len(data),VOCAB_SIZE))
    all_output_entropies = torch.zeros((len(data),VOCAB_SIZE))

    for i in tqdm(range(len(data))):
        point = data[i]
        context = point['context']
        entropies = get_entropy_for_context(context,layers_to_check=LAYERS_ENTROPIES)
        all_entropies[i,:] = entropies.detach()[:]

        entropies = get_entropy_for_context(context,layers_to_check=LAYERS_OUTPUTS)
        all_output_entropies[i,:] = entropies.detach()[:]
    
    torch.save(all_entropies.detach(), 'runs/all_entropies.pt')
    torch.save(all_output_entropies.detach(), 'runs/all_output_entropies.pt')
    print("Saved entropies.")
    

def add_0s_to_num(number):
    num_str = str(number)
    zeros_to_add = 3 - len(num_str)
    result = '0' * zeros_to_add + num_str
    
    return result

def get_context_length(context):
    input_ids = tokenizer(context).input_ids
    input_tokens = tokenizer.batch_decode(input_ids)
    # print(input_tokens)
    # print(input_tokens.index("\":"))
    return len(input_tokens)

def filter_low_prob_tokens(last_token_logits,last_token_probs,alpha,debug):
    if debug: print("ONE MINUS LAST TOKEN PROBS (MAX):",(1-torch.max(last_token_probs).item()))
    # indices_to_filter = torch.nonzero(last_token_probs<alpha*(1 - torch.max(last_token_probs))).squeeze()
    indices_to_filter = torch.nonzero(last_token_probs<alpha*torch.max(last_token_probs)).squeeze()
    indices_to_filter = torch.nonzero(last_token_probs<alpha*(1 - torch.max(last_token_probs))).squeeze()
    filtered_logits = last_token_logits[:]
    filtered_logits[indices_to_filter] = float('-inf')
    filtered_probs = F.softmax(filtered_logits).detach()
    if True: 
        print("FILTERED_PROBS MAXIMUM:",torch.max(filtered_probs))
        if torch.isnan(filtered_probs).any():
            print(filtered_logits.tolist())
            print(filtered_probs.tolist())

    assert not torch.isnan(filtered_probs).any()
    return filtered_probs

def scale_prob_by_entropy(id,data,all_entropies,all_output_entropies,beta,gamma,thresh,probs,debug):
    point = data[id]
    entropies = all_entropies[id].to("cuda")
    output_entropies = all_output_entropies[id].to("cuda")
    context = point['context']

    # subtract = entropies-math.log(get_context_length(id,inputs))
    # subtract = entropies - output_entropies
    # subtract = (entropies-math.log(get_context_length(id,inputs))) - \
    #     gamma*(output_entropies - math.log(get_context_length(id,inputs))) - 0.1*gamma
    eps = 1e-7
    if debug:
        print(f"ALERT: max(entropies-output_entropies)={torch.max(entropies-output_entropies)}")
        print(f"ALERT: math.log(L)={math.log(get_context_length(context))}")
    subtract = torch.where(math.log(get_context_length(context)) - output_entropies > thresh,-eps,(entropies-math.log(get_context_length(context)))-gamma*(output_entropies-math.log(get_context_length(context))))

    # Min entropy among those tokens whose probabilities are non-zero (exclude tokens which are filtered out)
    min_entropy_token = torch.argmin(torch.where(probs > 0.0, subtract, torch.tensor(float('inf')))).item()
    max_prior_prob_token = torch.argmax(probs).item()

    try:
        multiply = math.ceil(-1 - math.log10(-subtract[min_entropy_token].item()))
    except:
        subtract -= 0.1
        multiply = math.ceil(-1 - math.log10(-subtract[min_entropy_token].item()))
    subtract *= 10**(multiply) # Multiply full thing by a constant

    if debug: 
        print("Token with smallest entropy: ",min_entropy_token) # If this is 2982, it shows that "fur" is the most likely predicted according to entropy among the tokens that have high enough output probability
        print("Token with max probab: ",max_prior_prob_token)
        print("\nMaximum probability token (subtraction amt per beta):  ",subtract[max_prior_prob_token].item())
        print("Minimum entropy token (subtraction amt per beta): ",subtract[min_entropy_token].item())
        print()
        
        print("Maximum probability token (prior probab): ",max_prior_prob_token,probs[max_prior_prob_token].item())
        print("Minimum entropy token (prior probab): ",min_entropy_token,probs[min_entropy_token].item())
        print()
        
    logs = torch.log(probs)
    logs -= beta*subtract
    final_probs = F.softmax(logs,dim=0)

    if debug:
        print("Maximum probability token (posterior prob): ",max_prior_prob_token,final_probs[max_prior_prob_token].item())
        print("Minimum entropy token (posterior prob): ",min_entropy_token,final_probs[min_entropy_token].item())

    return final_probs

def entropy_penalised_decoding_integrated(id,data,debug=True,max_tokens=100,alpha=1e-2,beta=40,gamma=1,thresh=1e-4):
    all_entropies = torch.load("runs/all_entropies.pt")
    all_output_entropies = torch.load("runs/all_output_entropies.pt")
    device = torch.device("cuda")
    tokenizer.pad_token = "<s>"
    eos_token = tokenizer.eos_token_id

    queries = data[id]['queries']
    add_dict = {'intro':data[id]['intro'], 'context':data[id]['context'], 'queries':[]}

    for query in queries:
        prompt = format_template(data[id]['intro'], data[id]['context'], query)
        plain_prompt = format_template("", "", query)
        query_to_add = {'query':query}

        reg_out = regular_decoding(model,tokenizer,prompt,debug=False,max_tokens=max_tokens,show_tqdm=True, return_prob=False)
        cad_out = context_aware_decoding(model,tokenizer,prompt,plain_prompt,debug=False,alpha=0.5,max_tokens=max_tokens,show_tqdm=True)

        input_ids = tokenizer(prompt,return_tensors="pt",padding=True).input_ids.to(device)
        predicted_tokens = []

        for token in tqdm(range(max_tokens)):
            last_token_logits = model(input_ids).logits[0,-1,:]
            last_token_probs = F.softmax(last_token_logits)

            filtered_probs = filter_low_prob_tokens(last_token_logits,last_token_probs,alpha,debug).detach()
            final_probs = scale_prob_by_entropy(id,data,all_entropies,all_output_entropies,beta,gamma,thresh,filtered_probs,debug)

            max_index = torch.argmax(final_probs).item() # greedy decoding
            
            if max_index == eos_token:
                break

            predicted_tokens.append(max_index)
            
            input_ids = torch.cat([input_ids,torch.tensor([[max_index]]).to(device)],dim=1)
        
        print(predicted_tokens)
        ent_out = tokenizer.decode(predicted_tokens)

        print("prompt: ",prompt)
        print("output: ",ent_out)
        print()

        query_to_add['entropy_output'] = ent_out
        query_to_add['regular_output'] = reg_out
        query_to_add['cad_output'] = cad_out
        add_dict['queries'].append(query_to_add)

    return add_dict

def decode_all_inputs(data,max_tokens=100,alpha=1e-2,beta=40,gamma=1,thresh=1e-4):
    print("Decoding outputs...")
    outputs = []
    for id in tqdm(range(len(data))):
        add_dict = entropy_penalised_decoding_integrated(id,data,debug=False,max_tokens=max_tokens,alpha=alpha,beta=beta,gamma=gamma,thresh=thresh)
        outputs.append(add_dict)
    
    with open(f"runs/runs_regular_filtering/outputs_A_{alpha}_B_{beta}_G_{gamma}_T_{thresh:.0e}.json",'w') as file:
        json.dump(outputs,file,indent=4)
    
def run_pipeline():
    data = load_dataset()

    if GET_ENTROPIES:
        list_of_hooks = register_hooks("entropies")
        save_all_entropies(data)
        remove_hooks(list_of_hooks)
    
    if DECODE_INPUTS:
        decode_all_inputs(data,max_tokens=100,alpha=4e-2,beta=100,gamma=0,thresh=1)

        ## IMPORTANT!
        # ONE_MINUS_ALPHA_FILTERING: 0<alpha<=1: the smaller alpha is, the more likely we'll see contextual tokens which are ungrammatical or low in probability at the last token position
        #      at alpha=0, it allows any random low prob token to be used if it has low entropy. if alpha=1, it acts as a normal greedy decoding LLM 
        # REGULAR_FILTERING: alpha should be much smaller. The smaller alpha is, the more 'ungrammatical' words come in.
        # 1<=beta<100: the larger beta is, the more likely the contextual token with the minimum entropy will be favoured over memory
        # 0<=gamma<=1: at gamma = 0, it doesn't consider the output entropies; at gamma=1, it considers the difference between internal and output entropies;
        #       Keep gamma between 0 and 1.
        # 0<thresh<1: the smaller thresh is, the less likely it is to see contextual tokens with high probs at last 2 layers
        #       If thresh >= 0.1, it's likely that the output is simply (entropy-output_entropies) without any extra weightage on output penalisation
        #       At thresh < 0.01, you will probably have some threshold at which contextual tokens with high output activations do not force the LLM to follow them.

if __name__=="__main__":
    run_pipeline()