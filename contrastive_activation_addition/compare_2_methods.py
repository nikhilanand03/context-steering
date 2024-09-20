"""
Usage: python compare_2_methods.py --method1 "results/open_ended_scores/context-focus/results_layer=12_multiplier=3.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_open_ended_failures_fixed_geval_87%.json" --method2 "results/open_ended_scores/context-focus/contrastive+steering_open_ended_layer=12_mult=2.0.json"
Both files should correspond to the same questions in the same order (only different outputs)
"""

import json
import argparse
import random
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
import os

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


PROMPT = lambda d1,d2,b: f"""
You are a system designed to evaluate and compare two model outputs based on the following criteria:
- **Less Verbose**: Which output is more concise and avoids unnecessary details?
- **More Straightforward**: Which output directly answers the question without ambiguity or unnecessary complexity?
- **More Fluent**: Which output has better language flow, making it easier to read and understand?

Your task is to compare the following two outputs and decide which one is preferred based on these criteria. 
Provide a decision as 'Preferred: Method 1' or 'Preferred: Method 2'. 

Output 1: {d1['model_output'] if b else d2['model_output']}
Output 2: {d1['model_output'] if not b else d2['model_output']}
"""

def evaluate_outputs(prompt):
    messages = [
        (
            "system",
            "You are an preference evaluator for a language models' answers to questions. When given an instuction, question, and answer, you will decide the best answer based on the instruction. You will only ever return 'Preferred: Method 1' or 'Preferred: Method 2' and nothing else.",
        ),
        ("human", prompt),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content

def save_comparison(method1,method2):
    with open(method1,'r') as f:
        outputs1 = json.load(f)
    with open(method2,'r') as f:
        outputs2 = json.load(f)
    
    parent_dir1 = os.path.dirname(method1)
    parent_dir2 = os.path.dirname(method2)

    assert len(outputs1)==len(outputs2)
    assert parent_dir1==parent_dir2
    
    li = []
    for i in range(len(outputs1)):
        d1 = {"question":outputs1[i]["question"],"model_output":outputs1[i]["model_output"]}
        d2 = {"question":outputs2[i]["question"],"model_output":outputs2[i]["model_output"]}

        assert d1['question']==d2['question']

        preferred_output = evaluate_outputs(PROMPT(d1,d2,random.randint(0,1)))
        d = {"question":outputs1[i]["question"],"method1_output":outputs1[i]["model_output"],"method2_output":outputs2[i]["model_output"],"preferred":preferred_output}
        li.append(d)
    
    last_1,last_2 = method1.split("/")[-1],method2.split("/")[-1]
    new_file_path = os.path.join(parent_dir1, f"preferred_method1={last_1}_method2={last_2}.txt")
    with open(new_file_path,'w') as f:
        json.dump(li,f)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method1", type=str, required=True)
    parser.add_argument("--method2", type=str, required=True)
    args = parser.parse_args()

    save_comparison(args.method1,args.method2)