 import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.helpers import add_vector_from_position, add_vector_full_from_start, find_instruction_end_postion, get_model_path
from utils.tokenize import (
    tokenize_llama_chat,
    tokenize_llama_base,
    ADD_FROM_POS_BASE,
    ADD_FROM_POS_CHAT,
    ADD_FROM_POS_LATEST,
    PAD_TOKEN_LATEST,
    PAD_TOKEN_ID_LATEST,
    ADD_FROM_POS_GEMMA
)
from typing import Optional


class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.add_activations_full = None
        self.from_position = None
        self.ablate = False

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.calc_dot_product_with_after_steer = None
        self.dot_products = []
        self.dot_products_after_steer = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))

        if self.add_activations is not None:
            augmented_output = add_vector_from_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
                # ablate=self.ablate
            )
            output = (augmented_output,) + output[1:]
        elif self.add_activations_full is not None:
            augmented_output = add_vector_full_from_start(
                matrix=output[0],
                vector_full=self.add_activations_full
            )
            output = (augmented_output,) + output[1:]

        if self.calc_dot_product_with_after_steer is not None:
            last_token_aug_activations = augmented_output[0, -1, :]
            dot_product_aug = t.dot(last_token_aug_activations, self.calc_dot_product_with_after_steer) / (
                t.norm(last_token_aug_activations) * t.norm(self.calc_dot_product_with_after_steer)
            )
            self.dot_products_after_steer.append((0, dot_product_aug.cpu().item()))

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def setAblate(self,ablate):
        self.ablate = ablate

    def add(self, activations):
        self.add_activations = activations

    def add_full(self, activations_full):
        self.add_activations_full = activations_full

    def reset(self):
        self.add_activations = None
        self.add_activations_full = None
        self.ablate = False
        self.activations = None
        self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.calc_dot_product_with_after_steer = None
        self.dot_products = []
        self.dot_products_after_steer = []


class LlamaWrapper:
    def __init__(
        self,
        hf_token: str,
        size: str = "7b",
        use_chat: bool = True,
        override_model_weights_path: Optional[str] = None,
        use_latest: bool = False,
        use_mistral:bool = False
    ):
        self.device = "cuda:0" if t.cuda.is_available() else "cpu"
        self.use_chat = use_chat
        self.use_latest = use_latest
        self.use_mistral = use_mistral
        self.model_name_path = get_model_path(size, not use_chat, use_latest, use_mistral)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, token=hf_token
        )

        if self.model_name_path=="meta-llama/Meta-Llama-3.1-70B-Instruct":
            # model_config = {
            #     "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            #     "device_map": "auto",
            #     "torch_dtype": t.bfloat16
            # } ## If we wanna quantise
            model_config = {"device_map": "auto"} ## If we dont wanna quantise
        else:
            model_config = {}

        self.added_activations = False
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path, token=hf_token, **model_config
        )
        print("DEVICE",self.model.device,self.device)
        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))
        if size != "7b":
            self.model = self.model.half()

        if not self.model_name_path=="meta-llama/Meta-Llama-3.1-70B-Instruct":
            try:
                self.model = self.model.to(self.device)
            except (t.cuda.OutOfMemoryError,RuntimeError) as e:
                print(f"Out of Memory on {self.device}!")
                
                if t.cuda.device_count() > 1 and t.cuda.get_device_properties(1).total_memory > 0:
                    self.device = "cuda:1"
                    self.model = self.model.to(self.device)
                    print("Switched to cuda:1")
                else:
                    print("cuda:1 is not available")
                    raise e

        if self.model_name_path in ["meta-llama/Meta-Llama-3.1-8B-Instruct","meta-llama/Meta-Llama-3.1-70B-Instruct"]:
            self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_LATEST)[1:]).to(
                    self.device
                )
            self.tokenizer.pad_token_id = PAD_TOKEN_ID_LATEST
            # print(self.tokenizer.pad_token_id)
        elif self.model_name_path=="google/gemma-2b-it":
            self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_GEMMA)[1:]).to(
                    self.device
                )
        elif self.model_name_path=="mistralai/Mistral-7B-Instruct-v0.3":
            self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_CHAT)[1:]).to(
                    self.device
                )
        else:
            if use_chat:
                self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_CHAT)[1:]).to(
                    self.device
                )
            else:
                self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_BASE)[1:]).to(
                    self.device
                )
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.model.layers:
            layer.from_position = pos

    def generate(self, tokens, max_new_tokens=100):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            generated = self.model.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, top_k=1, pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.batch_decode(generated)[0]

    def generate_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None, max_new_tokens: int = 50) -> str:
        if self.use_chat:
            tokens = tokenize_llama_chat(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt
            )
        else:
            tokens = tokenize_llama_base(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output)
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def get_logits(self, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            try:
                logits = self.model(tokens).logits
            except:
                logits = self.model(tokens.to(self.model.device)).logits
            return logits

    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None) -> t.Tensor:
        if self.use_chat:
            tokens = tokenize_llama_chat(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt
            )
        else:
            tokens = tokenize_llama_base(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output)
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, ablate=False):
        self.added_activations = True
        self.model.model.layers[layer].add(activations)
        self.model.model.layers[layer].setAblate(ablate)
    
    def set_add_activations_full(self, layer, activations_full):
        self.added_activations = True
        self.model.model.layers[layer].add_full(activations_full)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def set_calc_dot_product_with_after_steer(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with_after_steer = vector

    def get_dot_products_after_steer(self, layer):
        return self.model.model.layers[layer].dot_products_after_steer

    def reset_all(self):
        self.added_activations = False
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
