import json
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

B_HEADER, E_HEADER = "<|start_header_id|>","<|end_header_id|>"
EOT_ID = "<|eot_id|>"

with open("hotpot_training_data.json",'r') as f:
    hotpot_data = json.load(f)

def template_llama_3_1_8B(
    documents: List[str],
    question: str
):
    system_prompt = "You are a Contextual QA Assistant. Use the following retrieved contexts to answer any questions that may follow."
    input_content = ""
    input_content += B_HEADER + "system" + E_HEADER + "\n\n" + system_prompt + EOT_ID

    context_input = ""
    i = 1
    for doc in documents:
        context_input += f"[Document {i}]: {doc}\n\n"
        i+=1

    input_content += f"{B_HEADER}user{E_HEADER}\n\n{context_input.strip()}{EOT_ID}\n{B_HEADER}assistant{E_HEADER}\n\n"
    input_content += f"Will do! I'll use these contexts to answer your questions.{EOT_ID}\n{B_HEADER}user{E_HEADER}\n\n{question.strip()}{EOT_ID}\n{B_HEADER}assistant{E_HEADER}\n\n"
        
    return input_content

# Load the LLaMA 3.1 8B model and tokenizer
model_name = "meta-llama/LLaMA-3.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

new_json_li = []

# print(template_llama_3_1_8B(["Steve Jobs' last name is Jobs.","Apple is the greatest company of the 2000s."],"What is Steve's last name?"))
for i in range(len(hotpot_data)):
    d = hotpot_data[i]
    print(len(d['context']))
    docs = ["".join(d['context'][i][1]) for i in range(len(d['context']))]
    prompt = template_llama_3_1_8B(docs,d['question'])
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)
    resp = tokenizer.decode(output[0], skip_special_tokens=True)
    
    d_new = {"question": d['question'], "answer": d['answer'], "model_output": resp}

    new_json_li.append(d_new)

assert len(new_json_li)==len(hotpot_data)

with open("hotpot_results.json",'w') as f:
    json.dump(new_json_li,f)

print('Saved results.')