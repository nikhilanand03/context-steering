import os
from tqdm import tqdm
import json

"""
Data format:
"question":"..."
"answers":["..",".."]
"ctxs":[ctx1,ctx2,..]
    ctx_i: {"title":"...","text":"..","has_answer":true/false,"is_gold":true/false}
"nq_annotated_gold": not-important

"""

ITEMS_PER_DS = 200

def generate_context(non_answer_texts,answer_text,insert_index):
    context = ""
    for i in range(len(non_answer_texts)+1):
        if i<insert_index: context+=non_answer_texts[i]+"\n\n"
        elif i==insert_index: context+=answer_text+"\n\n"
        else: context+=non_answer_texts[i-1]+"\n\n"
    return context.rstrip("\n")

def generate_prompt(id,num_contexts,insert_index,data_path): # type = 'irrel','mild_rel','rel'
    assert num_contexts<=30 and insert_index<num_contexts
    with open(data_path,'r') as file:
        lines = file.readlines()
        d = json.loads(lines[id])

    proper_question = d['question']

    all_contexts = d['ctxs']
    
    for i,context in enumerate(all_contexts):
        if(context['hasanswer']):
            answer_text = context['text']
            answer_i = i
        
    all_contexts = all_contexts[:answer_i] + all_contexts[answer_i+1:]
    
    non_answer_contexts = all_contexts[:num_contexts]
    non_answer_texts = [item['text'] for item in non_answer_contexts]

    proper_context = generate_context(non_answer_texts,answer_text,insert_index)

    # print(proper_context)
    
    proper_prompt = f"<Context>{proper_context}</Context><Question>{proper_question}</Question>"
    
    return proper_prompt,d['answers']

def save_dataset(num_contexts,data_path="nq-open-10_total_documents_gold_at_0.jsonl",fixed_index=None):
    to_save = []
    if fixed_index is not None:
        range_insert_ids = [fixed_index]
    else:
        range_insert_ids = range(num_contexts)

    for id in range(ITEMS_PER_DS//len(range_insert_ids)):
        for insert_index in range_insert_ids:
            item,answers = generate_prompt(id,num_contexts=num_contexts,insert_index=insert_index,data_path=data_path)
            try:
                context, question = item.split("</Context><Question>")[0][9:], item.split("</Context><Question>")[1][:-11]
                formatted_question = "Context: <P> " + context + "</P>\nQuestion: " + question
                to_save.append({"question":formatted_question,"answers":answers})
            except:
                print("Failed saving one item: ")
                print(item)
        
    with open(f"longcontexts2/test_dataset_open_ended_version=longcontext_num_contexts={num_contexts}.json",'w') as f:
        json.dump(to_save,f)

def run_pipeline():
    # save_dataset(3)
    # save_dataset(4)
    # save_dataset(5)
    # save_dataset(6)
    # save_dataset(7)
    # save_dataset(8)
    # save_dataset(9)
    # save_dataset(10)
    # save_dataset(13,"nq-open-30_total_documents_gold_at_0.jsonl")
    # save_dataset(15,"nq-open-30_total_documents_gold_at_0.jsonl")
    # save_dataset(17,"nq-open-30_total_documents_gold_at_0.jsonl")
    # save_dataset(19,"nq-open-30_total_documents_gold_at_0.jsonl")
    # save_dataset(21,"nq-open-30_total_documents_gold_at_0.jsonl")
    # save_dataset(24,"nq-open-30_total_documents_gold_at_0.jsonl")
    # save_dataset(27,"nq-open-30_total_documents_gold_at_0.jsonl")
    save_dataset(20,"nq-open-30_total_documents_gold_at_0.jsonl",fixed_index=10)

if __name__ == "__main__":
    run_pipeline()