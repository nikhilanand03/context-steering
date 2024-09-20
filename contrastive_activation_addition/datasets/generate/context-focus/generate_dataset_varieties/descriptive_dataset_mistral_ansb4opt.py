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

if model_name=="mistralai/Mistral-7B-Instruct-v0.3":
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

SYSTEM_PROMPT_MISTRAL = """You are an AI assistant. Answer the following question in one line.
Provide only the requested information without additional commentary.
"""

def tokenize_mistral_chat(user_input, context_answer):
    if model_name=="mistralai/Mistral-7B-Instruct-v0.3":
        input_content = B_INST + SYSTEM_PROMPT_MISTRAL + f"{user_input.strip()}{E_INST} {context_answer} However,"
        print(input_content)
        return tokenizer(input_content, return_tensors="pt").to(DEVICE)

def ask_gpt(answer,question):
    start_words1 = "Remember to start your answer with the words 'Based on the context,'."
    start_words2 = "Based on the context, "
    messages = [
        ("system",SYSTEM_PROMPT(start_words1,start_words2)),
        ("human", f"Question: {question}\nAnswer: {answer}"),
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content

def ask_mistral(question, context_answer):
    tokens = tokenize_mistral_chat(question, context_answer)
    generated_ids = model.generate(**tokens,max_new_tokens=65,pad_token_id=PAD_TOKEN_ID)
    output = tokenizer.batch_decode(generated_ids)[0].split(ADD_FROM_POS)[-1].strip(END_TOK).strip()
    return output

def get_context_answer(ques,context_short_ans):
    context_answer = ask_gpt(context_short_ans,ques)
    return context_answer

def process_data(data):

    li_of_dics = []
    count = 0
    MAX_ITEMS = 1500

    for d in tqdm(data):
        context_opt = d['answer_matching_behavior']
        question,choices = tuple(d['question'].split("\n\n"))
        split_word = "</"+question[question.index("<")+1:question.index(">")]+">"

        context,question = tuple(item for item in question.split(split_word))
        context = context + split_word
        question = question.strip()

        options = {"(A)": choices.split("\n")[1],"(B)": choices.split("\n")[2]}

        context_short_answer = options[context_opt][5:]
        context_answer = get_context_answer(question,context_short_answer)

        newOptionX = ask_mistral(question,context_answer)[len(context_answer)+1:]
        newOptionO = f"So, the answer is {context_short_answer}."

        prompt = f"Context: {context} \nQuestion: {question}\nAnswer: {context_answer}\n\nChoices:"

        if context_opt=="(A)":
            prompt += f"\n (A) {newOptionO}\n (B) {newOptionX}"
        elif context_opt=="(B)":
            prompt += f"\n (A) {newOptionX}\n (B) {newOptionO}"
        
        dic = {"question":prompt, "answer_matching_behavior":d['answer_matching_behavior'],"answer_not_matching_behavior":d['answer_not_matching_behavior']}

        li_of_dics.append(dic)
        print(dic)
        count += 1

        if count>MAX_ITEMS:
            break
    
    return li_of_dics

def save_data(data):
    with open("generate_dataset_mistral_ans_b4_opt.json",'w') as file:
        json.dump(data,file)

def run_pipeline():
    with open("generate_dataset_cleaned_contexts.json",'r') as file:
        data = json.load(file)

    data_processed = process_data(data)
    

    save_data(data_processed)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    run_pipeline()