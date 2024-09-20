"""
Use CAA to steer the model, inputting the context and question

Usage:
python prompt_model.py --context "CONTEXT" --question "QUESTION" --layer L --multiplier m --use_latest
Set layer to 13 and multiplier between -2 and 2 
"""

import argparse
from utils.tokenize import E_INST,ADD_FROM_POS_LATEST,ADD_FROM_POS_GEMMA
from llama_wrapper import LlamaWrapper
import torch
import os

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def get_vector_path(layer):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(dir)
    return os.path.join(script_dir, "1COMPLETED_RUNS", "FULL_ANALYSIS_Llama-3.1-8b", 
        "normalized_vectors", "context-focus", f"vec_layer_{layer}_Meta-Llama-3.1-8B-Instruct.pt")

def prompt_with_steering(
    model: LlamaWrapper,
    context: str,
    question: str
):
    input = f"Context: <P> {context} </P>\nQuestion: {question}"
    # input = f"{context}\n{question}"
    model_output = model.generate_text(
        user_input=input, max_new_tokens=350
    )
    if model.model_name_path=="google/gemma-2-2b-it":
        split_token = ADD_FROM_POS_GEMMA
    elif model.model_name_path=="meta-llama/Meta-Llama-3.1-8B-Instruct":
        split_token = ADD_FROM_POS_LATEST
    else:
        split_token = E_INST
    return model_output.split(split_token)[-1].strip().strip("<|eot_id|>")

def load_model(layer,multiplier,use_latest=True,use_mistral=False):
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
    vector = torch.load(get_vector_path(layer))
    vector = vector.to(model.device)
    model.set_add_activations(layer, multiplier * vector)

    return model

def change_multiplier(layer,multiplier,model):
    model.set_save_internal_decodings(False)
    model.reset_all()
    vector = torch.load(get_vector_path(layer))
    vector = vector.to(model.device)
    model.set_add_activations(layer, multiplier * vector)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--use_latest", action="store_true", default=False)
    parser.add_argument("--use_mistral", action="store_true", default=False)
    args = parser.parse_args()

    steering_model = load_model(args.layer,args.multiplier, args.use_latest, args.use_mistral)
    
    print(prompt_with_steering(steering_model, args.context,args.question))
    
