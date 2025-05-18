"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --model_size 7b --behaviors sycophancy
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat, tokenize_multi_context
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class ComparisonDataset(Dataset):
    def __init__(self, data_path, token, model_name_path, use_chat, multicontext=False, no_options=False, contrastive=False):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.model_name_path = model_name_path
        self.multicontext = multicontext
        self.no_options = no_options
        self.contrastive = contrastive

    def prompt_to_tokens(self, instruction, model_output):
        if self.contrastive:
            print("CONTRASTIVE GENERATION")

            positive_input = instruction['question']
            negative_input = instruction['no_context_question']

            p_tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=positive_input
            )
            n_tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=negative_input
            )

            return t.tensor(p_tokens).unsqueeze(0), t.tensor(n_tokens).unsqueeze(0)
        
        elif self.no_options:
            # in this case, model_output is None
            print("NO OPTIONS CASE : PROMPT_TO_TOKENS")

            context_input = "\n\n".join(
                f"[Document {i+1}]: {doc}" 
                for i, doc in enumerate(instruction['rag'])
            )
            query = instruction['query']

            positive_input = f"{instruction['system_prompt']}\n\n{context_input}\n\n{query}"
            negative_input = f"{query}"

            p_tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=positive_input
            )
            n_tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=negative_input
            )

            return t.tensor(p_tokens).unsqueeze(0), t.tensor(n_tokens).unsqueeze(0)

        elif self.multicontext:
            tokens = tokenize_multi_context(
                self.tokenizer, 
                instruction['rag'],
                instruction['question'],
                options = instruction['options'],
                model_output=model_output)
        else:
            if self.use_chat:
                tokens = tokenize_llama_chat(
                    self.tokenizer,
                    user_input=instruction,
                    model_output=model_output
                )
            else:
                tokens = tokenize_llama_base(
                    self.tokenizer,
                    user_input=instruction,
                    model_output=model_output,
                )
            
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # pre_out = "Ans: " if ("Llama-3" in self.model_name_path or "Mistral" in self.model_name_path) else ""
        
        if self.no_options or self.contrastive:
            p_tokens, n_tokens = self.prompt_to_tokens(item, None)
        elif self.multicontext:
            p_text = item["answer_matching_behavior"]
            n_text = item["answer_not_matching_behavior"]

            p_tokens = self.prompt_to_tokens(item, p_text)
            n_tokens = self.prompt_to_tokens(item, n_text)
        else:
            p_text = item["answer_matching_behavior"]
            n_text = item["answer_not_matching_behavior"]
            q_text = item["question"]

            p_tokens = self.prompt_to_tokens(q_text, p_text)
            n_tokens = self.prompt_to_tokens(q_text, n_text)
            
        return p_tokens, n_tokens

def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: List[str],
    model: LlamaWrapper,
    override_dataset: str,
    suffix: str,
    multicontext: bool,
    no_options: bool,
    contrastive: bool
):
    # print(get_vector_path(behavior, 1, model.model_name_path, suffix=suffix))
    # print(get_activations_path(behavior, 1, model.model_name_path, "neg", suffix=suffix))
    data_path = get_ab_data_path(behavior,test=False,override_dataset=override_dataset)
    if not os.path.exists(get_vector_dir(behavior,normalized=False,suffix=suffix)):
        os.makedirs(get_vector_dir(behavior,normalized=False,suffix=suffix))
    if save_activations and not os.path.exists(get_activations_dir(behavior,suffix=suffix)):
        os.makedirs(get_activations_dir(behavior,suffix=suffix))

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
        multicontext=multicontext,
        no_options=no_options,
        contrastive=contrastive
    )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -2, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            get_vector_path(behavior, layer, model.model_name_path, suffix=suffix),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos", suffix=suffix),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg", suffix=suffix),
            )

def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
    use_latest: bool=False,
    use_mistral:bool=False,
    override_dataset:str=None,
    suffix:str="",
    multicontext:bool=False,
    no_options:bool=False,
    contrastive:bool=False
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    behaviors: behaviors to generate vectors for
    """
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model, use_latest=use_latest, use_mistral=use_mistral
    )
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model, override_dataset, suffix, multicontext, no_options, contrastive
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--use_latest", action="store_true", default=False)
    parser.add_argument("--use_mistral",action="store_true",default=False)
    parser.add_argument("--override_dataset", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--multicontext", action="store_true", default=False)
    parser.add_argument("--no_options", action="store_true", default=False)
    parser.add_argument("--contrastive", action="store_true", default=False)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors,
        args.use_latest,
        args.use_mistral,
        args.override_dataset,
        args.suffix,
        args.multicontext,
        args.no_options,
        args.contrastive
    )
