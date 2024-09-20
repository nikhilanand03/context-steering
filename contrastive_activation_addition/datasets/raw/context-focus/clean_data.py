import os
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm

# Initialize the AzureChatOpenAI instance
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Read the system prompt
with open("prompt1.txt", 'r') as f:
    SYSTEM_PROMPT = f.read()

def process_json_entry(json_entry):
    # Convert JSON entry to a string if it's not already
    if isinstance(json_entry, dict):
        json_str = json.dumps(json_entry, indent=2)
    else:
        json_str = json_entry

    # Create messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Please process the following JSON entry:\n\n{json_str}")
    ]

    # Get response from GPT
    response = llm.invoke(messages)
    
    return response.content

# Function to process a batch of JSON entries
def process_batch(json_entries):
    results = []
    for entry in tqdm(json_entries):
        result = process_json_entry(entry)
        print(result)
        try:
            results.append(eval(result))
        except:
            print(f"Problematic output: {result}")
            continue
    return results

def remove_non_P_tags(dataset):
    new_li = []
    for d in dataset:
        if d["org_context"][:3]=="<P>":
            new_li.append(d)
    return new_li

if __name__ == "__main__":
    # Load your JSON entries
    with open("raw_data_v1.json", 'r') as f:
        dataset = json.load(f)

    print(len(dataset))
    new_li = remove_non_P_tags(dataset)
    print(len(new_li))

    # Process a batch of entries (adjust the slice as needed)
    processed_entries = process_batch(new_li[-750:])  # Process first 10 entries

    # Save processed entries
    with open("processed_entries.json", 'w') as f:
        json.dump(processed_entries, f, indent=2)

    print(f"Processed {len(processed_entries)} entries.")