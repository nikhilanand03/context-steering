from typing import List
from transformers import PreTrainedTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

B_TEXT = "<|begin_of_text|>"
B_HEADER, E_HEADER = "<|start_header_id|>","<|end_header_id|>"
EOT_ID = "<|eot_id|>"

B_TEXT_GEMMA = "<bos>"
TURN_ST, TURN_E = "<start_of_turn>","<end_of_turn>"
EOT_ID_GEMMA = "<eos>"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_BASE = BASE_RESPONSE
ADD_FROM_POS_LATEST = f"{B_HEADER}assistant{E_HEADER}\n\n"
ADD_FROM_POS_GEMMA = f"{TURN_ST}assistant\n"

PAD_TOKEN_LATEST = "<|finetune_right_pad_id|>"
PAD_TOKEN_ID_LATEST = 128004

def template_mistral(
    user_input: str,
    model_output: str = None,
    system_prompt: str = None
):
    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    input_content += f"{B_INST} {user_input.strip()} {E_INST}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return input_content

def template_gemma(
    user_input: str,
    model_output: str = None,
    system_prompt: str = None
):
    input_content = ""
    if system_prompt is not None:
        input_content += TURN_ST + "user\n" + system_prompt + "\n" + f"{user_input.strip()}{TURN_E}\n{TURN_ST}assistant\n"
    else:
        input_content += f"{TURN_ST}user\n{user_input.strip()}{TURN_E}\n{TURN_ST}assistant\n"
    if model_output is not None:
        input_content += f" {model_output.strip()}"

    return input_content

def template_llama_3_1_8B(
    user_input: str,
    model_output: str = None,
    system_prompt: str = None
):
    input_content = ""
    if system_prompt is not None:
        input_content += B_HEADER + "system" + E_HEADER + "\n\n" + system_prompt + EOT_ID
    input_content += f"{B_HEADER}user{E_HEADER}\n\n{user_input.strip()}{EOT_ID}\n{B_HEADER}assistant{E_HEADER}\n\n"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
        
    return input_content

def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None
) -> List[int]:
    print(tokenizer.name_or_path)
    if tokenizer.name_or_path=="mistralai/Mistral-7B-Instruct-v0.3":
        input_content = template_mistral(user_input, model_output, system_prompt)
    elif tokenizer.name_or_path=="google/gemma-2-2b-it":
        input_content = template_gemma(user_input, model_output, system_prompt)
    elif tokenizer.name_or_path in ["meta-llama/Meta-Llama-3.1-8B-Instruct","meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        input_content = template_llama_3_1_8B(user_input, model_output, system_prompt)
        print(input_content)
    else: # Most prompt templates are like mistral, including Llama-2's template
        input_content = template_mistral(user_input, model_output, system_prompt)
    return tokenizer.encode(input_content)

def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)

def tokenize_multi_context(
    tokenizer: PreTrainedTokenizer, 
    documents: List[str], 
    question: str, 
    options: List[str] = None,
    model_output: str = None,
    system_add: str = ""
) -> List[int]:
    system_prompt = "You are a Contextual QA Assistant. Use the following retrieved contexts to answer any questions that may follow." + f" {system_add}"
    header = f"{B_HEADER}system{E_HEADER}\n\n{system_prompt}{EOT_ID}"
    
    context_input = "\n\n".join(
        f"[Document {i+1}]: {doc}" 
        for i, doc in enumerate(documents)
    )

    if options is not None:
        question = f"{question}\n\nOptions:\n (A) {options[0]}\n (B) {options[1]}"
    
    user_context = f"{B_HEADER}user{E_HEADER}\n\n{context_input.strip()}{EOT_ID}"
    assistant_reply = f"{B_HEADER}assistant{E_HEADER}\n\nWill do! I'll use these contexts to answer your questions.{EOT_ID}"
    final_question = f"{B_HEADER}user{E_HEADER}\n\n{question.strip()}{EOT_ID}\n{B_HEADER}assistant{E_HEADER}\n\n"
    
    output_token_string = f"{header}{user_context}{assistant_reply}{final_question}"

    if model_output is not None:
        output_token_string += f" {model_output.strip()}"
    
    print(output_token_string)

    return tokenizer.encode(output_token_string)