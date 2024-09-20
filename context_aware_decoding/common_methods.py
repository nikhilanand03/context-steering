import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output
# from datasets import load_dataset
from tqdm import tqdm
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()


def load_memotrap(): # returns (context_inputs,plain_inputs,context_outputs)
    with open("data/memotrap_dataset.txt",'r') as file:
        dataset_string = file.readlines()

    arr = [item.split("\t")[:2] for item in dataset_string]
    context_inputs = [item[0] for item in arr]
    # outputs = [eval(item[1]) for item in arr]

    assert len(context_inputs)==215

    context_outputs = []
    for inp in context_inputs:
        first_quote = inp.find('"')
        second_quote = inp[first_quote+1:].find('"') + first_quote + 1
        context_output = inp[first_quote+1:second_quote]
        context_outputs.append(context_output)

    assert len(context_outputs)==215

    plain_inputs = []
    for i in range(len(context_inputs)):
        plain_input = context_inputs[i][:13]+context_inputs[i][context_inputs[i].find(":"):]
        plain_inputs.append(plain_input)

    assert len(plain_inputs)==215

    return context_inputs,plain_inputs,context_outputs

def load_technews():
    ds_path = "ds_tech_news"
    dataset = {}
    
    for filename in os.listdir(ds_path):
        file_path = os.path.join(ds_path, filename)
        
        if filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                contents = file.read()
                cqa = contents.split("</context>")
                context = cqa[0][9:]
                qa = cqa[1].split("</a>")[:-1]
                qa_new = [item.split("<a>") for item in qa]
                qa_final = [[item[0].strip("\n").lstrip("<q>").rstrip("</q>"),item[1]] for item in qa_new]
                # print(qa_final)
                # print("***")
                dataset[context]=qa_final
                
    return dataset

def remove_words(ans,ques): # ques's words are removed from ans
    filtered_sentence = []

    words_to_remove = set()
    for word in nltk.word_tokenize(ques):
        lemma = lemmatizer.lemmatize(word.lower(),pos="v")
        words_to_remove.add(lemma)

    stop_words = set(stopwords.words('english'))
    for word in nltk.word_tokenize(ans):
        lemma = lemmatizer.lemmatize(word.lower(),pos="v")
        word = word.rstrip('.,!?;:')
        if lemma not in words_to_remove:
            filtered_sentence.append(word)

    return ' '.join(filtered_sentence)

def get_accuracy(regular_outs,method_outs,true_outs):
    regular_score,method_score = 0,0
    working_ids = []
    n = len(regular_outs)
    
    for i in range(n):
        reg = False
        if regular_outs[i]==true_outs[i]:
            regular_score+=1
            reg = True
        if method_outs[i]==true_outs[i]:
            method_score+=1
            if not reg:
                working_ids.append(i)
        
    print(f"Regular Decoding Correct: {100*regular_score/n}%")
    print(f"Special Decoding Correct: {100*method_score/n}%")
    return working_ids

def regular_decoding(model,tokenizer,prompt,debug=True,
                     max_tokens=1,show_tqdm=True, return_prob=False): # ID is the index within the inputs list
    if debug: print("prompt: ",prompt)
    device = torch.device("cuda")
    tokenizer.pad_token = "<s>"
    eos_token = tokenizer.eos_token_id
    input_ids = tokenizer(prompt,return_tensors="pt",padding=True).input_ids.to(device)
    predicted_tokens = []

    token_iterator = tqdm(range(max_tokens)) if (max_tokens>1 and show_tqdm) else range(max_tokens)
    
    for token in token_iterator:
        last_token_logits = model(input_ids).logits[0,-1,:]
        last_token_probs = F.softmax(last_token_logits)

        max_index = torch.argmax(last_token_probs).item() # greedy decoding
        
        if max_index == eos_token:
            break

        predicted_tokens.append(max_index)
        
        input_ids = torch.cat([input_ids,torch.tensor([[max_index]]).to(device)],dim=1)
    
    out = tokenizer.decode(predicted_tokens) if max_tokens>1 \
                else tokenizer.batch_decode(predicted_tokens)[0]
    if debug: print("output: ",out)
    if not debug:
        if return_prob==True and max_tokens==1:
            return last_token_probs
        else:
            return out