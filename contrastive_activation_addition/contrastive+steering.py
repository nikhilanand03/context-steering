"""
Use CAA to steer the model, inputting the context and question

Usage:
python prompt_model.py --context "CONTEXT" --question "QUESTION" --layer L --multiplier m --use_latest
Set layer to 13 and multiplier between -2 and 2 
"""

import argparse
from utils.tokenize import E_INST,ADD_FROM_POS_LATEST
from llama_wrapper import LlamaWrapper
import torch
from utils.tokenize import tokenize_llama_chat,EOT_ID,EOT_ID_GEMMA
from utils.helpers import get_model_path
import os
import torch.nn.functional as F
from tqdm import tqdm
import json

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def get_vector_path(layer,model_name_path):
    # dir = os.path.dirname(os.path.abspath(__file__))
    # print(dir)
    model_name = model_name_path.split("/")[-1]
    return f"normalized_vectors/context-focus/vec_layer_{layer}_{model_name}.pt"

# def prompt_with_steering(
#     model: LlamaWrapper,
#     context: str,
#     question: str
# ):
#     input = f"Context: <P> {context} </P>\nQuestion: {question}"
#     model_output = model.generate_text(
#         user_input=input, max_new_tokens=100
#     )
#     split_token = E_INST if not model.use_latest else ADD_FROM_POS_LATEST
#     return model_output.split(split_token)[-1].strip().strip("<|eot_id|>")

def load_model(layer,multiplier,use_latest,use_mistral):
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size="7b",
        use_chat=True,
        override_model_weights_path=None,
        use_latest = use_latest,
        use_mistral = use_mistral
    )
    model.set_save_internal_decodings(False)
    model.reset_all()
    vector = torch.load(get_vector_path(layer,get_model_path(size="7b", is_base=False, use_latest=use_latest, use_mistral=use_mistral)))
    vector = vector.to(model.device)
    model.set_add_activations(layer, multiplier * vector)

    return model

def toggle_steering(model, layer, multiplier):
    # print(model.added_activations)
    if model.added_activations:
        model.set_save_internal_decodings(False)
        model.reset_all()
    elif not model.added_activations:
        model.set_save_internal_decodings(False)
        model.reset_all()
        vector = torch.load(get_vector_path(layer,get_model_path(size="7b", is_base=False, use_latest=model.use_latest, use_mistral=model.use_mistral)))
        vector = vector.to(model.device)
        model.set_add_activations(layer, multiplier * vector)
    # print(model.added_activations)
    
    return model

def filter_logits(logits, probs, threshold):
    if threshold==0:
        return logits
    
    mask = probs < threshold
    neg_inf = torch.full_like(logits, float('-inf'))
    filtered_logits = torch.where(mask, neg_inf, logits)

    return filtered_logits
    
def cadsteer_generate(model,input,max_tokens,layer,multiplier,alpha=2,t=None):
    # input = f"Context: <P> {context} </P>\nQuestion: {question}"
    print(input)
    input_tokens = tokenize_llama_chat(tokenizer=model.tokenizer, user_input=input)
    input_tokens = torch.tensor(input_tokens).unsqueeze(0).to(model.device)
    # print(input_tokens)
    generated_tokens = []

    for i in range(max_tokens):
        # print(f"ITER{i}")
        model = toggle_steering(model,layer,multiplier) # Toggle to unsteered model
        # print(f"Unsteered model - Input tokens shape: {input_tokens.shape}")
        logits_plain = model.get_logits(input_tokens)[0,-1:,:]
        plain_probs = F.softmax(logits_plain,dim=-1)
        # print(f"Unsteered logits shape: {logits_plain.shape}")

        model = toggle_steering(model,layer,multiplier) # Toggle to steered model
        # print(f"Steered model - Input tokens shape: {input_tokens.shape}")
        logits_steered = model.get_logits(input_tokens)[0,-1:,:]
        steered_probs = F.softmax(logits_steered,dim=-1)
        # print(f"Steered logits shape: {logits_steered.shape}")

        if alpha:
            logits_net = (1+alpha)*logits_steered - alpha*logits_plain
        else:
            logits_net = logits_steered - logits_plain

        # print(t*torch.max(logits_steered) if t else 0.0)
        filtered_net_logits = filter_logits(logits_net,plain_probs,t*torch.max(steered_probs) if t else 0.0)

        probs_net = F.softmax(filtered_net_logits,dim=-1)

        max_token = torch.argmax(probs_net).unsqueeze(0).unsqueeze(0).to(model.device)

        if model.model_name_path in ["meta-llama/Meta-Llama-3.1-8B-Instruct","meta-llama/Meta-Llama-3.1-70B-Instruct"]:
            eos_token = EOT_ID
        elif model.model_name_path=="google/gemma-2-2b-it":
            eos_token = EOT_ID_GEMMA
        elif model.model_name_path=="mistralai/Mistral-7B-Instruct-v0.3":
            eos_token = "</s>"
        else:
            eos_token = "</s>"

        if max_token.item()==model.tokenizer.convert_tokens_to_ids(eos_token):
            break
        generated_tokens.append(max_token.item())
    
        input_tokens = torch.cat((input_tokens, max_token), dim=1).to(model.device)
        # print(f"New input tokens shape: {input_tokens.shape}")
    
    print(model.tokenizer.decode(generated_tokens))
    return model.tokenizer.decode(generated_tokens)

def run_cadsteer(data_path,mult,layer,alpha=2,t=1e-3,type="open_ended",use_latest=True,use_mistral=False):
    data_list = []
    model = load_model(13,mult,use_latest=use_latest,use_mistral=use_mistral) # Initialises to a steering model
    with open(data_path,'r') as f:
        data = json.load(f)
        for item in tqdm(data):
            input = item['question']
            output = cadsteer_generate(model,input,65 if type=="open_ended" else 400,layer,mult,alpha,t)
            data_list.append({'question': input, 'model_output': output})
    
    with open(f"results/context-focus/contrastive+steering_open_ended_layer={layer}_mult={mult}_type={type}_alpha={alpha}_thresh={t}_usemistral={use_mistral}.json",'w') as f:
        json.dump(data_list,f)

def get_data_path(type):
    if type=="open_ended":
        return "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_failures_fixed_geval_87%.json"
    elif type=="if_eval":
        return "datasets/test/if-eval/if_eval_prompts.json"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--type", type=str, choices=["open_ended", "if_eval"], default="open_ended")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--thresh", type=float, required=True)
    parser.add_argument("--use_mistral", action="store_true", default=False)

    args = parser.parse_args()

    use_latest = not args.use_mistral

    alpha = args.alpha # Set to None if you want it to run the infinite alpha decoding
    thresh = args.thresh # The smaller this is, the fewer tokens are filtered out
    # It does not use alpha during saving, so we fix alpha for each run and we make the plots, move them elsewhere before trying a new value of alpha
    
    for m in args.multipliers:
        print(f"Mult={m}; Layer={args.layers[0]}")
        run_cadsteer(get_data_path(args.type),mult=m,layer=args.layers[0],alpha=alpha,t=thresh,type=args.type,use_latest=use_latest,use_mistral=args.use_mistral)
    # run_cadsteer("datasets/test/context-focus/test_dataset_open_ended.json",mult=1)