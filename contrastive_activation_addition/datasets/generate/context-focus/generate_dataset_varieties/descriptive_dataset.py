"""
Usage: python descriptive_dataset.py (useful for any model)
Usage: python descriptive_dataset.py --type "however" (used for the mistral_however dataset)

You need to hardcode the following:
- Changing model name
- Changing input and output file names (mainly output filename)
"""

import json
import os
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from tqdm import tqdm
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

B_TEXT = "<|begin_of_text|>"
B_HEADER, E_HEADER = "<|start_header_id|>","<|end_header_id|>"
EOT_ID = "<|eot_id|>"
ADD_FROM_POS_LATEST = f"{B_HEADER}assistant{E_HEADER}\n\n"
PAD_TOKEN_LATEST = "<|finetune_right_pad_id|>"
PAD_TOKEN_ID_LATEST = 128004

B_TEXT_GEMMA = "<bos>"
TURN_ST, TURN_E = "<start_of_turn>","<end_of_turn>"
EOT_ID_GEMMA = TURN_E
ADD_FROM_POS_GEMMA = f"{TURN_ST}assistant\n"
PAD_TOKEN_ID_GEMMA = 0
PAD_TOKEN_GEMMA = "<pad>"

B_TEXT_MISTRAL = "<s>"
B_INST, E_INST = "[INST]", "[/INST]"
END_TOK_MISTRAL = "</s>"
ADD_FROM_POS_MISTRAL = E_INST
PAD_TOKEN_ID_MISTRAL = 0
PAD_TOKEN_MISTRAL = "<pad>"

DEVICE = "cuda:1"

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "google/gemma-2-2b-it"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if model_name=="meta-llama/Meta-Llama-3.1-8B-Instruct":
    END_TOK = EOT_ID
    ADD_FROM_POS = ADD_FROM_POS_LATEST
    PAD_TOKEN_ID = PAD_TOKEN_ID_LATEST
    tokenizer.pad_token = PAD_TOKEN_LATEST
elif model_name=="google/gemma-2-2b-it":
    END_TOK = EOT_ID_GEMMA
    ADD_FROM_POS = ADD_FROM_POS_GEMMA
    PAD_TOKEN_ID = PAD_TOKEN_ID_GEMMA
    tokenizer.pad_token = PAD_TOKEN_GEMMA
elif model_name=="mistralai/Mistral-7B-Instruct-v0.3":
    END_TOK = END_TOK_MISTRAL
    ADD_FROM_POS = ADD_FROM_POS_MISTRAL
    PAD_TOKEN_ID = PAD_TOKEN_ID_MISTRAL
    tokenizer.pad_token = PAD_TOKEN_MISTRAL

model = AutoModelForCausalLM.from_pretrained(model_name).eval()
model.to(DEVICE)

SYSTEM_PROMPT = lambda start_words1, start_words2: f"""
You are a helpful assistant that converts a question and short answer into a complete sentence response.
{start_words1}

Example:
Question: Who plays Shaun Murphy in the good doctor?
Answer: Freddie Highmore
Your Output: {start_words2}Freddie Highmore plays Shaun Murphy in the Good Doctor.

Guidelines:
1. Provide only a single, complete sentence as your output. {start_words1}
3. Incorporate the question and answer into your response without adding extra information.
4. Maintain the facts as given, even if they appear incorrect.
5. Do not include any explanations or additional commentary.
"""

SYSTEM_PROMPT_LLAMA = """You are an AI assistant. Answer the following question in one line.
Provide only the requested information without additional commentary.
"""

SYSTEM_PROMPT_MISTRAL = """You are an AI assistant. Answer the following question in two sentences. Do not exceed the sentence limit.
Provide only the requested information without additional commentary.
"""

def tokenize_llama_chat(user_input,context_text):
    if model_name=="meta-llama/Meta-Llama-3.1-8B-Instruct":
        input_content = B_HEADER + "system" + E_HEADER + "\n\n" + SYSTEM_PROMPT_LLAMA + EOT_ID
        # input_content = ""
        input_content += f"{B_HEADER}user{E_HEADER}\n\n{user_input.strip()}{EOT_ID}\n{B_HEADER}assistant{E_HEADER}\n\nNotwithstanding the context,"
        # print(input_content)
        return tokenizer(input_content, return_tensors="pt").to(DEVICE)
    elif model_name=="google/gemma-2-2b-it":
        input_content = TURN_ST + "user\n" + SYSTEM_PROMPT_LLAMA + f"{user_input.strip()}{TURN_E}\n{TURN_ST}assistant\nNotwithstanding the context,"
        print(input_content)
        return tokenizer(input_content, return_tensors="pt").to(DEVICE)
    elif model_name=="mistralai/Mistral-7B-Instruct-v0.3" and context_text is not None:
        input_content = B_INST + SYSTEM_PROMPT_MISTRAL + f"{user_input.strip()}{E_INST} {context_text} However, it's important to note"
        print(input_content)
        return tokenizer(input_content, return_tensors="pt").to(DEVICE)
    elif model_name=="mistralai/Mistral-7B-Instruct-v0.3":
        input_content = B_INST + SYSTEM_PROMPT_MISTRAL + f"{user_input.strip()}{E_INST} Notwithstanding the context,"
        print(input_content)
        return tokenizer(input_content, return_tensors="pt").to(DEVICE)

def ask_gpt(answer,question, context):
    start_words1 = "Remember to start your answer with the words 'Based on the context,'." if context else "Remember to start your answer with the words 'The context states that'."
    start_words2 = "Based on the context, " if context else "The context states that "
    messages = [
        ("system",SYSTEM_PROMPT(start_words1,start_words2)),
        ("human", f"Question: {question}\nAnswer: {answer}"),
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content

def ask_llama(question,context_text):
    tokens = tokenize_llama_chat(question,context_text)
    generated_ids = model.generate(**tokens,max_new_tokens=65,pad_token_id=PAD_TOKEN_ID)
    output = tokenizer.batch_decode(generated_ids)[0].split(ADD_FROM_POS)[-1].strip(END_TOK).strip()
    return output

def is_named_entity(word):
    tagged = pos_tag([word])
    chunked = ne_chunk(tagged)
    return isinstance(chunked[0], Tree)

def get_new_option(i,optionA,optionB,question,context_answer,type):
    letter_str = optionA[:5] if i==0 else optionB[:5]
    context_letter_str = " "+context_answer+" "

    context_option = optionA if context_answer=="(A)" else optionB
    context_option_text = context_option.split(context_letter_str)[1]
    is_context = context_letter_str==letter_str

    if is_context:
        new_option_text = ask_gpt(context_option_text,question,is_context)
    else:
        if model_name=="mistralai/Mistral-7B-Instruct-v0.3" and type=="however":
            context_option_full_text = ask_gpt(context_option_text,question,is_context)
        else:
            context_option_full_text = None
        new_option_text = ask_llama(question,context_option_full_text)
    
    return letter_str+new_option_text

def process_data(data,type):

    li_of_dics = []
    count = 0
    MAX_ITEMS = 1500

    for d in tqdm(data):
        context_answer = d['answer_matching_behavior']
        question,choices = tuple(d['question'].split("\n\n"))
        split_word = "</"+question[question.index("<")+1:question.index(">")]+">"
        context,question = tuple(item for item in question.split(split_word))
        context = context + split_word
        options = [choices.split("\n")[1],choices.split("\n")[2]]
        # print(context,"**",question,"**",options)

        for i in range(len(options)):
            options[i] = get_new_option(i,options[0],options[1],question,context_answer,type)

        prompt = ""
        prompt += f"Context: {context} \nQuestion: {question}\n\nChoices:\n{options[0]}\n{options[1]}"
        
        dic = {"question":prompt, "answer_matching_behavior":d['answer_matching_behavior'],"answer_not_matching_behavior":d['answer_not_matching_behavior']}
        li_of_dics.append(dic)
        print(dic)
        count += 1
        if count>MAX_ITEMS:
            break
    
    return li_of_dics

def save_data(data):
    with open("generate_dataset_llama_induce_output_mistral_however.json",'w') as file:
        json.dump(data,file)
    # with open("generate_dataset_llama_induce_output_mistral.json",'w') as file:
    #     json.dump(data,file)
    # with open("generate_dataset_llama_induce_output.json",'w') as file:
    #     json.dump(data,file)

def run_pipeline(type):
    with open("generate_dataset_cleaned_contexts.json",'r') as file:
        data = json.load(file)

    data_processed = process_data(data,type)

    save_data(data_processed)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["notwithstanding","however"],default="notwithstanding")
    args = parser.parse_args()

    run_pipeline(args.type)