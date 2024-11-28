"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b
"""

import json
from llama_wrapper import LlamaWrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from torch import Tensor
import torch as t
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST,ADD_FROM_POS_LATEST,ADD_FROM_POS_GEMMA, tokenize_llama_chat, tokenize_multi_context
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_steering_vector_from_path,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_if_eval_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def process_item_ab(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
    multicontext: bool=False,
    no_options:bool=False
) -> Dict[str, str]:
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    question: str = item["question"]

    if not multicontext:
        model_output = model.get_logits_from_text(
            user_input=question, model_output="(", system_prompt=system_prompt
        )
    else:
        tokens = tokenize_multi_context(
            model.tokenizer, 
            item['rag'], 
            question, 
            item['options'],
            model_output="(")
        tokens = t.tensor(tokens).unsqueeze(0).to(model.device)
        model_output = model.get_logits(tokens)

    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    
    response = {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }

    print("DEBUG (A,B) PROBS: ",a_prob, b_prob)

    if multicontext:
        response["options"] = item["options"]
        response["rag"] = item["rag"]

    return response

def process_item_open_ended(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
    multicontext: bool=False,
    no_options: bool=False
) -> Dict[str, str]:
    question = item["question"]

    if multicontext:
        tokens = tokenize_multi_context(
            model.tokenizer, 
            item['rag'], 
            question,
            system_add="Please restrict your response to one sentence.")
        tokens = t.tensor(tokens).unsqueeze(0).to(model.device)
        model_output = model.generate(tokens, max_new_tokens=100)
    elif no_options:
        context_input = "\n\n".join(
            f"[Document {i+1}]: {doc}" 
            for i, doc in enumerate(item['rag'])
        )
        query = item['question']
        sys_prompt = "You are a Contextual QA Assistant. Please answer the following question according to the given context. Please restrict your response to one sentence. "

        input_text = f"{sys_prompt}\n\nContext:\n\n{context_input}\n\nQuestion: {query}"

        model_output = model.generate_text(
            user_input=input_text, max_new_tokens=400
        )
    else:
        model_output = model.generate_text(
            user_input=question, system_prompt=system_prompt, max_new_tokens=100
        )

    if model.model_name_path=="google/gemma-2-2b-it":
        split_token = ADD_FROM_POS_GEMMA
    elif model.model_name_path in ["meta-llama/Meta-Llama-3.1-8B-Instruct","meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        split_token = ADD_FROM_POS_LATEST
    else:
        split_token = E_INST
    print(split_token)
    print(model_output.split(split_token)[-1].strip())

    response = {
        "question": question,
        "model_output": model_output.split(split_token)[-1].strip(),
        "raw_model_output": model_output,
    }

    if multicontext: 
        response['rag'] = item['rag']
        response['answer'] = item['answer']

    return response

def process_item_open_ended_dynamic(
    item: Dict[str,str], # the json item containing "question"
    model: LlamaWrapper,
    layer: int,
    vector: Tensor
) -> Dict[str, str]:
    question = item["question"]
    tokens = tokenize_llama_chat(tokenizer=model.tokenizer, user_input=question)
    tokens = t.tensor(tokens).unsqueeze(0).to(model.device) # will be a tensor of shape (1,num_tokens)

    num_tokens = tokens.shape[1]
    hidden_size = model.model.config.hidden_size
    current_activations_full = t.zeros((1, num_tokens-1, hidden_size)).to(model.device)
    vector_sh = vector.unsqueeze(0).unsqueeze(0).to(model.device) # Shape (1, 1, hidden_size)

    # This will be added directly to the state. So it needs to be changed every time the token state size changes.

    max_new_tokens = 100

    for i in range(max_new_tokens):
        print("current_activations_full.shape:",current_activations_full.shape)
        current_activations_full_0 = t.cat((current_activations_full, 0*vector_sh), dim=1).to(model.device)
        current_activations_full_2 = t.cat((current_activations_full, 2*vector_sh), dim=1).to(model.device)

        # Next token probs with zero-steering
        model.reset_all()
        model.set_add_activations_full(layer, current_activations_full_0)
        logits_0 = model.get_logits(tokens)
        last_token_logits_0 = logits_0[0, -1, :]
        last_token_probs_0 = t.softmax(last_token_logits_0, dim=-1).to(model.device)
        
        # Next token probs with 2-steering
        model.reset_all()
        model.set_add_activations_full(layer, current_activations_full_2)
        logits_2 = model.get_logits(tokens)
        last_token_logits_2 = logits_2[0, -1, :]
        last_token_probs_2 = t.softmax(last_token_logits_2, dim=-1).to(model.device)

        # Getting optimal steering multiplier
        last_token_probs_0 = last_token_probs_0 / last_token_probs_0.sum()
        last_token_probs_2 = last_token_probs_2 / last_token_probs_2.sum()
        kl_div = t.sum(last_token_probs_0 * t.log(last_token_probs_0 / last_token_probs_2)).to(model.device)
        steer_mult = min(3*kl_div.item(),2)

        print("3*KL Divergence: ",3*kl_div.item())

        current_activations_full = t.cat((current_activations_full, steer_mult*vector_sh), dim=1).to(model.device)

        print(f"Setting add activations full to one with shape {current_activations_full.shape} and doing forward pass.")
        model.reset_all()
        model.set_add_activations_full(layer, current_activations_full)
        logits = model.get_logits(tokens)
        last_token_logits = logits[0, -1, :]
        last_token_probs = t.softmax(last_token_logits, dim=-1).to(model.device)
        max_index = t.argmax(last_token_probs)
        new_token = t.tensor([[max_index]])
        tokens = t.cat((tokens, new_token.to(model.device)), dim=1)
    
    model_output = model.tokenizer.decode(tokens[0])

    if model.model_name_path=="google/gemma-2-2b-it":
        split_token = ADD_FROM_POS_GEMMA
    elif model.model_name_path in ["meta-llama/Meta-Llama-3.1-8B-Instruct","meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        split_token = ADD_FROM_POS_LATEST
    else:
        split_token = E_INST

    print(split_token)
    print(model_output.split(split_token)[-1].strip())

    return {
        "question": question,
        "model_output": model_output.split(split_token)[-1].strip(),
        "raw_model_output": model_output,
    }

def process_item_if_eval(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
    multicontext: bool=False,
    no_options:bool=False
) -> Dict[str, str]:
    question = item["question"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=400
    )
    split_token = E_INST if not model.use_latest else ADD_FROM_POS_LATEST
    print(split_token)
    print(model_output.split(split_token)[-1].strip())
    return {
        "question": question,
        "model_output": model_output.split(split_token)[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_tqa_mmlu(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = model.get_logits_from_text(
        user_input=prompt, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_dir(settings.behavior,suffix=settings.suffix)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        # "truthful_qa": process_item_tqa_mmlu,
        # "mmlu": process_item_tqa_mmlu,
        "if_eval": process_item_if_eval
    }

    test_datasets = {
        "ab": get_ab_test_data(settings.behavior,settings.override_ab_dataset),
        "open_ended": get_open_ended_test_data(settings.behavior, settings.override_oe_dataset_path),
        # "truthful_qa": get_truthful_qa_data(),
        # "mmlu": get_mmlu_data(),
        "if_eval": get_if_eval_data()
    }
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
        use_latest = settings.use_latest,
        use_mistral = settings.use_mistral
    )
    print(settings.use_latest,settings.use_mistral)
    print(model.model)
    print(model.tokenizer)
    
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    print("TOKEN IDS (A,B): ", a_token_id,b_token_id)
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]
    
    print("TEST_DATA[0]",test_data[0])

    for layer in layers:
        if settings.override_vector_path is not None:
            vector = get_steering_vector_from_path(settings.override_vector_path)
        else:
            name_path = model.model_name_path
            if settings.override_vector_model is not None:
                name_path = settings.override_vector_model
            if settings.override_vector is not None:
                vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
            elif settings.override_vector_behavior is not None: # Using a vector from a different behaviour
                vector = get_steering_vector(settings.override_vector_behavior, layer, name_path, normalized=True)
            else:
                vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True, suffix=settings.suffix)

        if settings.model_size != "7b":
            vector = vector.half()
        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                if settings.dynamic_m and settings.type=="open_ended":
                    result = process_item_open_ended_dynamic(
                        item=item,
                        model=model,
                        layer=layer,
                        vector=vector
                    )
                else:
                    model.reset_all()
                    model.set_add_activations(
                        layer, multiplier * vector, 
                        # settings.ablate
                    )
                    result = process_methods[settings.type](
                        item=item,
                        model=model,
                        system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                        a_token_id=a_token_id,
                        b_token_id=b_token_id,
                        multicontext=settings.multicontext,
                        no_options=settings.no_options
                    )

                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--ablate", action="store_true", default=False)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa", "mmlu", "if_eval"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--override_vector_behavior", type=str, default=None)
    parser.add_argument("--override_vector_path", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--use_latest", action="store_true", default=False)
    parser.add_argument("--use_mistral", action="store_true", default=False)
    parser.add_argument("--override_oe_dataset_path", type=str, default=None)
    parser.add_argument("--override_ab_dataset", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--dynamic_m", action="store_true", default=False)
    parser.add_argument("--multicontext", action="store_true", default=False)
    parser.add_argument("--no_options", action="store_true", default=False)

    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.override_vector_behavior = args.override_vector_behavior
    steering_settings.override_vector_path = args.override_vector_path
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path
    steering_settings.use_latest = args.use_latest
    steering_settings.use_mistral = args.use_mistral
    steering_settings.override_oe_dataset_path = args.override_oe_dataset_path
    steering_settings.override_ab_dataset = args.override_ab_dataset
    steering_settings.suffix = args.suffix
    steering_settings.ablate = args.ablate
    steering_settings.dynamic_m = args.dynamic_m
    steering_settings.multicontext = args.multicontext
    steering_settings.no_options = args.no_options

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        test_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            overwrite=args.overwrite,
        )