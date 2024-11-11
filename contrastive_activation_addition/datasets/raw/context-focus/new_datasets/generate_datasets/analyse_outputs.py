import json
from tqdm import tqdm

filename = "hotpot_results_correct_3k.json"
with open(filename,'r') as f:
    data = json.load(f)

with open("scoring_prompt.txt",'r') as f:
    FAITHFULNESS_SYSTEM_PROMPT = f.read()

from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def score_json_entry(json_entry):
    try:
        del json_entry['raw_model_output']
    except:
        pass
    json_str = json.dumps(json_entry, indent=2)

    # Create messages
    messages = [
        SystemMessage(content=FAITHFULNESS_SYSTEM_PROMPT),
        HumanMessage(content=f"Please process the following JSON entry:\n\n{json_str}")
    ]

    # Get response from GPT
    response = llm.invoke(messages)
    
    return response.content

mistakes = []

scores = [[]]
i = 0
for row in tqdm(data):
    for item in row:
        score = score_json_entry({"question": item['question'],
            "ground_truth_answer": item['answer'],
            "model_output": item['model_output']})

        # score = '1' if item['answer'] in item['model_output'] else '0'
        item['score'] = score
        scores[-1].append(item)
        print(item)

        if score=='0':
            mistakes.append(item)
        
        i+=1

        if i%200==0:
            with open(f"scores_{filename}",'w') as f:
                json.dump(scores,f)

            with open(f"mistakes_{filename}",'w') as f:
                json.dump(mistakes,f)

    scores.append([])

with open(f"scores_{filename}",'w') as f:
    json.dump(scores,f)

with open(f"mistakes_{filename}",'w') as f:
    json.dump(mistakes,f)