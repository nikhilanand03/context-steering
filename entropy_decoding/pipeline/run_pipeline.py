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
SAVE_PLOTS = False
GET_PREDS = False
GET_FAILURES = True
GET_ENTROPIES = False
DECODE_INPUTS = False
MEMOTRAP_SCORE = False
VOCAB_SIZE = 32000

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

def get_prediction(id,inputs,context_outputs,memory_outputs):
    input_token_list = tokenizer.batch_decode(tokenizer(inputs[id]).input_ids)
    
    out = regular_decoding(model,tokenizer,inputs[id],debug=False,max_tokens=1,show_tqdm=True, return_prob=False)

    attrs = {'prompt':inputs[id],'context_output':context_outputs[id],'memory_output':memory_outputs[id],'cont_correct?':0,'mem_correct?':0} # populate with keys {"prompt":"",output:"","correct":True/False,'correct_pred':True/False}

    wordC = context_outputs[id]
    wordM = memory_outputs[id]
    wordC_tok,wordM_tok = tokenizer(wordC).input_ids[1],tokenizer(wordM).input_ids[1]

    last_contok_ind = input_token_list.index("\":")
    
    layers_to_check = [22,24,26,28]
    intokens_to_check = list(range(last_contok_ind+1)) # These are the token positions at the input.
    attrs['intokens'] = [input_token_list[i] for i in intokens_to_check]

    actC = False
    actM = False

    for layer, intoken in product(layers_to_check,intokens_to_check):
        act = F.softmax(model.lm_head(model.model.norm(activations[layer][0,intoken,:])),dim=0)

        # print(f"Layer {layer}, Intoken {intoken}")
        top_values, top_indices = torch.topk(act, k=50)
        
        if wordC_tok in top_indices:
            actC=True
            attrs[f'top_tokens_C{layer},{intoken}'] = tokenizer.batch_decode(top_indices.tolist())
        if wordM_tok in top_indices:
            actM = True
            attrs[f'top_tokens_M{layer},{intoken}'] = tokenizer.batch_decode(top_indices.tolist())
    
    attrs['cont_correct?'] = actC
    attrs['mem_correct?'] = actM

    return attrs

def get_entropy_across_intokens(matrix):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    
    device = matrix.device
    epsilon = 1e-10
    p_log_p = matrix * torch.log(matrix + epsilon)

    entropies = -torch.sum(p_log_p, dim=0)
    print(entropies.shape)

    return entropies

def get_entropy_for_id(id,inputs,context_outputs,memory_outputs,layers_to_check):
    context_token_list = tokenizer.batch_decode(tokenizer(inputs[id]).input_ids)
    context_token_list = context_token_list[:context_token_list.index('":')+1]

    print(context_token_list)
    
    out=regular_decoding(model,tokenizer,inputs[id],debug=False,max_tokens=1,show_tqdm=True, return_prob=False)

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

    random_id = tokenizer(memory_outputs[id]).input_ids[1]
    print(f"Entropies for id = {id}")
    print(f"Context output '{context_outputs[id]}' entropy: {min_ent[tokenizer(context_outputs[id]).input_ids[1]]}")
    print("Printing random consecutive entropies now.")
    print(f"{tokenizer.batch_decode([random_id])} entropy: {min_ent[random_id]}")
    print(f"{tokenizer.batch_decode([random_id+1])} entropy: {min_ent[random_id+1]}")
    print(f"{tokenizer.batch_decode([random_id+2])} entropy: {min_ent[random_id+2]}")
    print(f"{tokenizer.batch_decode([random_id+3])} entropy: {min_ent[random_id+3]}")

    # Great, entropies look accurate! Most entropies are around the same value since most tokens are just uniform everywhere.
    # Only 'fur's entropy is significantly higher due to visible activations.

    return min_ent

def save_all_entropies(inputs,context_outputs,memory_outputs):
    print("Saving entropies...")
    all_entropies = torch.zeros((len(inputs),VOCAB_SIZE))
    all_output_entropies = torch.zeros((len(inputs),VOCAB_SIZE))

    for id in tqdm(range(len(inputs))):
        entropies = get_entropy_for_id(id,inputs,context_outputs,memory_outputs,layers_to_check=LAYERS_ENTROPIES)
        all_entropies[id,:] = entropies.detach()[:]

        entropies = get_entropy_for_id(id,inputs,context_outputs,memory_outputs,layers_to_check=LAYERS_OUTPUTS)
        all_output_entropies[id,:] = entropies.detach()[:]
    
    torch.save(all_entropies.detach(), 'runs/all_entropies.pt')
    torch.save(all_output_entropies.detach(), 'runs/all_output_entropies.pt')
    print("Saved entropies.")

def get_all_predictions(inputs,context_outputs,memory_outputs):
    list_of_attrs = []
    print("Getting predictions")
    for id in tqdm(range(len(inputs))):
        attrs = get_prediction(id,inputs,context_outputs,memory_outputs)
        list_of_attrs.append(attrs.copy())
    
    try:
        with open("runs/predictions.json", 'w', encoding='utf-8') as f:
            json.dump(list_of_attrs, f, ensure_ascii=False, indent=4)
        print("Saved predictions to runs/predictions.json")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    

def show_prediction_accuracies():
    try:
        with open("runs/predictions.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Data successfully loaded from runs/predictions.json")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return
    
    cont_true,cont_false,mem_true,mem_false = 0,0,0,0

    for point in data:
        cont_correct = point['cont_correct?']
        mem_correct = point['mem_correct?']
        
        if cont_correct:
            cont_true += 1
        else:
            cont_false += 1

        if mem_correct:
            mem_true += 1
        else:
            mem_false += 1
    
    print(f"\tTrue\tFalse\nContext\t{cont_true}\t{cont_false}\nMemory\t{mem_true}\t{mem_false}")

def get_failures():
    print("Getting failures...")
    with open("runs/predictions.json",'r') as file:
        data = json.load(file)
    
    output = []
    
    for i in range(len(data)):
        point = data[i]
        cont_correct = point['cont_correct?']
        mem_correct = point['mem_correct?']

        if not cont_correct:
            point['id'] = i
            point = {'id':point['id'],'prompt':point['prompt'],'context_output':point['context_output'],
                     'memory_output':point['memory_output'],'cont_correct?':point['cont_correct?'],'mem_correct?':point['mem_correct?']}
            output.append(point)
        elif mem_correct:
            point['id'] = i
            point = {'id':point['id'],'prompt':point['prompt'],'context_output':point['context_output'],
                     'memory_output':point['memory_output'],'cont_correct?':point['cont_correct?'],'mem_correct?':point['mem_correct?']}
            output.append(point)
    
    print(f"Finished getting {len(output)} failures")

    try:
        with open("runs/failure_preds.json", 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print("Saved predictions to runs/predictions.json")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    
def save_plot_for_id(id,inputs,context_outputs,memory_outputs):
    input_token_list = tokenizer.batch_decode(tokenizer(inputs[id]).input_ids)
    
    out=regular_decoding(model,tokenizer,inputs[id],debug=False,max_tokens=1,show_tqdm=True, return_prob=False)
    
    # wordC = context_outputs[id]
    # wordM = memory_outputs[id]
    # wordO = out
    # wordSUM = 'sum'
    # wordWORD = 'word'
    wordQUOTE = 'quote'
    wordWORDS = 'words'
    wordQ = '‘'

    # wordC_tok,wordM_tok,wordO_tok,wordSUM_tok,wordWORD_tok = tokenizer(wordC).input_ids[1], tokenizer(wordM).input_ids[1],\
# tokenizer(wordO).input_ids[1],tokenizer(wordSUM).input_ids[1],tokenizer(wordWORD).input_ids[1]
    wordQUOTE_tok = tokenizer(wordQUOTE).input_ids[1]
    wordWORDS_tok = tokenizer(wordWORDS).input_ids[1]
    wordQ_tok = tokenizer(wordQ).input_ids[1]

    # matrixC = torch.zeros((len(layer_list),len(input_token_list)))
    # matrixM = torch.zeros((len(layer_list),len(input_token_list)))
    # matrixO = torch.zeros((len(layer_list),len(input_token_list)))
    # matrixSUM = torch.zeros((len(layer_list),len(input_token_list)))
    # matrixWORD = torch.zeros((len(layer_list),len(input_token_list)))
    matrixQUOTE = torch.zeros((len(layer_list),len(input_token_list)))
    matrixWORDS = torch.zeros((len(layer_list),len(input_token_list)))
    matrixQ = torch.zeros((len(layer_list),len(input_token_list)))
    
    for l in range(len(layer_list)):
        for i in range(len(input_token_list)):
            layer = layer_list[l]
            act = F.softmax(model.lm_head(model.model.norm(activations[layer][0,i,:])),dim=0)
            # probC = act[wordC_tok]
            # probM = act[wordM_tok]
            # probO = act[wordO_tok]
            # probSUM = act[wordSUM_tok]
            # probWORD = act[wordWORD_tok]
            probQUOTE = act[wordQUOTE_tok]
            probWORDS = act[wordWORDS_tok]
            probQ = act[wordQ_tok]
            
            # matrixC[l,i] = probC
            # matrixM[l,i] = probM
            # matrixO[l,i] = probO
            # matrixSUM[l,i] = probSUM
            # matrixWORD[l,i] = probWORD
            matrixQUOTE[l,i] = probQUOTE
            matrixWORDS[l,i] = probWORDS
            matrixQ[l,i] = probQ

    # sf_matM = F.softmax(matrixM,dim=1).detach()
    # sf_matC = F.softmax(matrixC,dim=1).detach()
    # sf_matO = F.softmax(matrixO,dim=1).detach()
    # sf_matSUM = F.softmax(matrixSUM,dim=1).detach()
    # sf_matWORD = F.softmax(matrixWORD,dim=1).detach()
    sf_matQUOTE = F.softmax(matrixQUOTE,dim=1).detach()
    sf_matWORDS = F.softmax(matrixWORDS,dim=1).detach()
    sf_matQ = F.softmax(matrixQ,dim=1).detach()

    # sf_matM_ = sf_matM[torch.arange(sf_matM.size(0)-1, -1, -1)]
    # sf_matC_ = sf_matC[torch.arange(sf_matC.size(0)-1, -1, -1)]
    # sf_matO_ = sf_matO[torch.arange(sf_matO.size(0)-1, -1, -1)]
    # sf_matSUM_ = sf_matSUM[torch.arange(sf_matSUM.size(0)-1, -1, -1)]
    # sf_matWORD_ = sf_matWORD[torch.arange(sf_matWORD.size(0)-1, -1, -1)]
    sf_matQUOTE_ = sf_matQUOTE[torch.arange(sf_matQUOTE.size(0)-1, -1, -1)]
    sf_matWORDS_ = sf_matWORDS[torch.arange(sf_matWORDS.size(0)-1, -1, -1)]
    sf_matQ_ = sf_matQ[torch.arange(sf_matQ.size(0)-1, -1, -1)]

    layer_list_ = layer_list[::-1]
    input_token_list_ = input_token_list[:]

    # to_plot = ['M','C','O','SUM','WORD','QUOTE','WORDS']
    to_plot = ['QUOTE','WORDS','Q'] # change this list if you want to plot other vars

    for plot in to_plot:
        plot_mat_ = sf_matQUOTE_ if plot=='QUOTE' else (sf_matWORDS_ if plot=='WORDS' else sf_matQ_) # change this condition if you want to plot other vars
        word_shown = wordQUOTE if plot=='QUOTE' else (wordWORDS if plot=='WORDS' else wordQ) #  change this condition if you want to plot other vars
        naming = 'memory' if plot=='M' else ('context' if plot=='C' else 'output')

        plt.figure(figsize=(10, 8))
        sns.heatmap(plot_mat_, cmap='coolwarm', fmt='g',
                    xticklabels=input_token_list_, yticklabels=layer_list_)
        
        plt.title(f'Matrix for the output "{word_shown}"')
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        
        plt.savefig(f'runs/plots/id{add_0s_to_num(id)}_{naming}_{word_shown}.png')
        plt.close()
    
def add_0s_to_num(number):
    num_str = str(number)
    zeros_to_add = 3 - len(num_str)
    result = '0' * zeros_to_add + num_str
    
    return result

def save_all_plots(inputs,context_outputs,memory_outputs):
    print("SAVING ALL PLOTS")
    for id in tqdm(range(len(inputs))):
        save_plot_for_id(id,inputs,context_outputs,memory_outputs)

def get_context_length(id,inputs):
    input_ids = tokenizer(inputs[id]).input_ids
    input_tokens = tokenizer.batch_decode(input_ids)
    # print(input_tokens)
    # print(input_tokens.index("\":"))
    return input_tokens.index("\":")+1

def filter_low_prob_tokens(last_token_logits,last_token_probs,alpha):
    print("ONE MINUS LAST TOKEN PROBS (MAX):",(1-torch.max(last_token_probs).item()))
    indices_to_filter = torch.nonzero(last_token_probs<alpha*(1 - torch.max(last_token_probs))).squeeze()
    filtered_logits = last_token_logits[:]
    filtered_logits[indices_to_filter] = float('-inf')
    filtered_probs = F.softmax(filtered_logits)
    return filtered_probs

def scale_prob_by_entropy(id,inputs,context_outputs,memory_outputs,all_entropies,all_output_entropies,beta,thresh,probs,debug):
    wordC,wordM = context_outputs[id],memory_outputs[id]
    wordC_token = tokenizer(wordC).input_ids[1]
    wordM_token = tokenizer(wordM).input_ids[1]
    if debug: 
        print(f"Prob (context token) = {probs[wordC_token]}")
        print(f"Context: {wordC},{wordC_token}\nMemory: {wordM},{wordM_token}\n")

    entropies = all_entropies[id].to("cuda")
    output_entropies = all_output_entropies[id].to("cuda")

    # subtract = entropies-math.log(get_context_length(id,inputs))
    # subtract = entropies - output_entropies
    # subtract = (entropies-math.log(get_context_length(id,inputs))) - \
    #     gamma*(output_entropies - math.log(get_context_length(id,inputs))) - 0.1*gamma
    eps = 1e-7
    print(f"ALERT: max(entropies-output_entropies)={torch.max(entropies-output_entropies)}")
    print(f"ALERT: math.log(L)={math.log(get_context_length(id,inputs))}")
    subtract = torch.where(math.log(get_context_length(id,inputs)) - output_entropies > thresh,-eps,entropies-output_entropies)

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

def entropy_penalised_decoding(id,inputs,context_outputs,memory_outputs,debug=True,alpha = 1e-1,beta=30,thresh=1e-9):
    all_entropies = torch.load("runs/all_entropies.pt")
    all_output_entropies = torch.load("runs/all_output_entropies.pt")
    device = torch.device("cuda")
    tokenizer.pad_token = "<s>"
    eos_token = tokenizer.eos_token_id
    input_ids = tokenizer(inputs[id],return_tensors="pt",padding=True).input_ids.to(device)
    last_token_logits = model(input_ids).logits[0,-1,:].detach()
    last_token_probs = F.softmax(last_token_logits,dim=0)
    
    
    ## PRINTING THE WORDS THAT HAVE DEVIANT ENTROPIES
    # if debug:
    #     values, counts = torch.unique(all_entropies[id], return_counts=True)
    #     mode = values[torch.argmax(counts)].item()
    #     print(f"Mode:{mode}")
    #     words = torch.nonzero(all_entropies[id] < round(mode,4)).squeeze().tolist()
    #     words_strings = tokenizer.batch_decode(words)
    #     for word in words_strings:
    #         print(word,end=", ")
    #     print("\nDone")
    
    ## FILTER OUT LOW PROBABILITY TOKENS AND THEN SCALE ACCORDING TO ENTROPIES
    filtered_probs = filter_low_prob_tokens(last_token_logits,last_token_probs,alpha).detach()
    final_probs = scale_prob_by_entropy(id,inputs,context_outputs,memory_outputs,all_entropies,all_output_entropies,beta,thresh,filtered_probs,debug)

    out = tokenizer.batch_decode([torch.argmax(final_probs).tolist()])[0]
    return out

def decode_all_inputs(inputs,plain_inputs,context_outputs,memory_outputs,alpha=1e-2,beta=30,thresh=1e-9):
    print("Decoding outputs...")
    output_dicts = []
    for id in tqdm(range(215)):
        out_epd = entropy_penalised_decoding(id,inputs,context_outputs,memory_outputs,debug=True,alpha=alpha,beta=beta,thresh=thresh)
        out_reg = regular_decoding(model,tokenizer,inputs[id],debug=False,max_tokens=1,show_tqdm=True, return_prob=False)
        out_cad = context_aware_decoding(model,tokenizer,inputs[id],plain_inputs[id],debug=False,alpha=0.5,max_tokens=1,show_tqdm=True)

        dic = {'prompt':inputs[id], 'context_output':context_outputs[id], 'memory_output':memory_outputs[id],
               'entropy_output':out_epd, 'regular_output':out_reg, 'CAD output': out_cad}
        output_dicts.append(dic)
    
    try:
        with open("runs/memotrap_decoded_outputs.json", 'w', encoding='utf-8') as f:
            json.dump(output_dicts, f, ensure_ascii=False, indent=4)
        print("Saved predictions to runs/memotrap_decoded_outputs.json")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    
    print("Decoded all prompts and saved.")

def get_memotrap_score():
    with open("runs/memotrap_decoded_outputs.json", 'r') as file:
        data = json.load(file)
    
    reg_score = 0
    ent_score = 0
    cad_score = 0
    
    for point in data:
        cont_op = point['context_output']
        ent_op = point['entropy_output']
        reg_op = point['regular_output']
        cad_op = point['CAD output']

        if cont_op.startswith(ent_op):
            ent_score+=1
        if cont_op.startswith(reg_op):
            reg_score+=1
        if cont_op.startswith(cad_op):
            cad_score+=1
        
    print(f"Regular decoding score: {reg_score/len(data)}")
    print(f"Entropy decoding score: {ent_score/len(data)}")
    print(f"Context-aware decoding score: {cad_score/len(data)}")
    
def run_pipeline():
    inputs,plain_inputs,context_outputs,memory_outputs = load_memotrap()

    if SAVE_PLOTS:
        list_of_hooks = register_hooks("plots")
        save_all_plots(inputs,context_outputs,memory_outputs)
        remove_hooks(list_of_hooks)

    if GET_PREDS:
        list_of_hooks = register_hooks("preds")
        get_all_predictions(inputs,context_outputs,memory_outputs)
        remove_hooks(list_of_hooks)
        show_prediction_accuracies()
    
    if GET_FAILURES:
        get_failures()

    if GET_ENTROPIES:
        list_of_hooks = register_hooks("entropies")
        save_all_entropies(inputs,context_outputs,memory_outputs)
        remove_hooks(list_of_hooks)
    
    if DECODE_INPUTS:
        decode_all_inputs(inputs,plain_inputs,context_outputs,memory_outputs,alpha=0.01,beta=40,thresh=1e-4)
        ## IMPORTANT!
        # alpha: the smaller alpha is, the more likely we'll see contextual tokens which are ungrammatical or low in probability at the last token position
        # beta: the larger beta is, the more likely the contextual token with the minimum entropy will be favoured over memory
        # thresh: the smaller thresh is, the less likely it is to see contextual tokens with high probs at last 2 layers
    
    if MEMOTRAP_SCORE:
        get_memotrap_score()

if __name__=="__main__":
    run_pipeline()