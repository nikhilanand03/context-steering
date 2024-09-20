from tqdm import tqdm
from common_methods import *
from context_decoder import ContextDecoder
import json
from pynvml import *

DATASET_PATH = "datasets/final_dataset_both_prompts.json"

def print_device_info():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print("******")
    print(f'total    : {info.total/(1024**3)} GB')
    print(f'free     : {info.free/(1024**3)} GB')
    print(f'used     : {info.used/(1024**3)} GB')
    print("******")

print_device_info()
decoder = ContextDecoder("mistralai/Mistral-7B-Instruct-v0.2")
print("Loaded model")
print_device_info()

def get_prompt(data):
    ctxs = data['ctxs']
    question = data['question']
    c_idx = get_true_ctx_index(ctxs)
    c_false = (ctxs[:c_idx]+ctxs[c_idx+1:])[:15]
    # print(len(c_false))
    
    assert ctxs[c_idx]['hasanswer'] and not c_false[0]['hasanswer'] and not c_false[1]['hasanswer']
    
    # docs = "\n\nDocument [1]: (" + c_false[0]['text']+")" + "\n\nDocument [2]: (" + ctxs[c_idx]['text'] + ")\n\nDocument [3]: (" + c_false[1]['text'] + ")"
    
    docs = ""
    for i in range(len(c_false)+1):
        # print(i)
        pos = (len(c_false)+1)//4 + 2
        
        docs = docs+f"\n\nDocument [{i}]: ("
        if i==pos:
            docs = docs+ctxs[c_idx]['text']
        elif i<pos:
            docs = docs+c_false[i]['text']
        else:
            docs = docs+c_false[i-1]['text']
            
    docs=docs+")"
    
    prompt = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Keep the answers short and to the point." + docs + "\n\nQuestion: " + question + "?"+"\nAnswer:"

    prompt_nc_wo_docs =  "\n\nWrite a high-quality answer for the given question. You do not need to provide answers as per the documents above and may use your own discretion. Keep the answers short and to the point." + "\n\nQuestion: " + question + "?"+"\nAnswer:"
    
    return prompt,prompt_nc_wo_docs

def get_true_ctx_index(ctxs):
    for i in range(len(ctxs)):
        ctx=ctxs[i]
        # print(ctx)
        if ctx['hasanswer']:
            return i

def make_dataset(paths = ["datasets/nq-open-10_total_documents_gold_at_0.jsonl",\
                 "datasets/nq-open-10_total_documents_gold_at_4.jsonl"]):
    lines = []

    for path in paths:
        with open(path,'r') as file:
            lines = lines + file.readlines()
    
    to_save = []

    nfp = []
    for i in range(len(lines)):
        json_string=lines[i]
        data = json.loads(json_string)

        try:
            prompt,prompt_nc_wo_docs = get_prompt(data)
        except:
            nfp+=[i]
        answers = data['answers']
        long_answer = data['nq_annotated_gold']['long_answer']
        short_answers = data['nq_annotated_gold']['short_answers']
    
        dict_to_add = {"prompt":prompt, "negative_prompt_wo_docs":prompt_nc_wo_docs,
                       "short_answers":short_answers, "long_answer":long_answer}
        to_save.append(dict_to_add)

    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(to_save, f, ensure_ascii=False, indent=4)
    print("Saved dataset")

def load_data():
    try:
        with open(DATASET_PATH, 'r') as file:
            data = json.load(file)
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        
    return data

def generate_completions(alpha=2.5,max_tokens=50,negative=False):
    data = load_data()
    data_filtered = data[:1000]
    
    for i in range(len(data_filtered)):
        print(f"{i}/{len(data_filtered)}")
        point = data_filtered[i]
        prompt = point['prompt']
        if not negative: prompt_nc = prompt.split("\n\nQuestion: ")[-1]
        else:
            docs = prompt[prompt.index("to the point.")+13:prompt.index("\n\nQuestion:")]
            prompt_nc = docs+ point['negative_prompt_wo_docs']
            # print(prompt_nc)
        # prompt_nc
        
        with torch.no_grad():
            # print_device_info()
            print("Regular Decoding")
            out_base = decoder.regular_decoding(prompt,max_tokens=max_tokens,debug=False,show_tqdm=True)
            
            # print_device_info()
            print("CAD")
            out_cad = decoder.context_aware_decoding(prompt,prompt_nc,alpha=alpha,max_tokens=max_tokens,
                                                     debug=False,show_tqdm=True)
            # print_device_info()
        
        point['baseline_completion'] = out_base
        point['cad_completion'] = out_cad
        print(prompt.split("\n\nQuestion: ")[-1],point['short_answers'],"***",out_base,"***",out_cad,"***")
        

    with open("runs/completions_negative_prompting.json", 'w', encoding='utf-8') as f:
        json.dump(data_filtered, f, ensure_ascii=False, indent=4)

    print("Generated and saved completions")

def pipeline():
    # 1. Makes the dataset and saves to a path
    make_dataset()

    # 2. Generate and save completions
    generate_completions(negative=True)

if __name__=="__main__":
    pipeline()