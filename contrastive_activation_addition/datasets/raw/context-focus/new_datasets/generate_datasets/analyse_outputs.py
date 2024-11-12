import json
import sys
from tqdm import tqdm

start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

filename = "hotpot_results_100k.json"
output_filename = f"scores_{start_idx}_{end_idx}_{filename}"
mistakes_filename = f"mistakes_{start_idx}_{end_idx}_{filename}"

with open(filename, 'r') as f:
    data = json.load(f)

data = data[start_idx:end_idx] if end_idx is not None else data[start_idx:]

with open("scoring_prompt.txt", 'r') as f:
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
)

def score_json_entry(json_entry):
    try:
        del json_entry['raw_model_output']
    except:
        pass
    json_str = json.dumps(json_entry, indent=2)

    messages = [
        SystemMessage(content=FAITHFULNESS_SYSTEM_PROMPT),
        HumanMessage(content=f"Please process the following JSON entry:\n\n{json_str}")
    ]

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

        item['score'] = score
        scores[-1].append(item)
        print(item)

        if score == '0':
            mistakes.append(item)
        
        i += 1

        if i % 200 == 0:
            with open(output_filename, 'w') as f:
                json.dump(scores, f)
            with open(mistakes_filename, 'w') as f:
                json.dump(mistakes, f)

    scores.append([])

# Save the final results
with open(output_filename, 'w') as f:
    json.dump(scores, f)

with open(mistakes_filename, 'w') as f:
    json.dump(mistakes, f)