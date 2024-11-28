import json

with open("hotpot_training_data.json", 'r') as f:
    data_tr = json.load(f)[2300*8:2700*8] # CHANGE THIS RANGE

supporting_facts_dataset = []

i = 0
for item in data_tr:
    supp_facts = item['supporting_facts']
    contexts = item['context']

    matched_contexts = []
    for fact in supp_facts:
        fact_title = fact[0]
        for context in contexts:
            if context[0] == fact_title:
                matched_contexts.append("".join(context[1]))

    new_item = {
        'question': item['question'],
        'answer': item['answer'],
        'supporting_contexts': matched_contexts
    }

    supporting_facts_dataset.append(new_item)

print(supporting_facts_dataset[0])  # Print the first item to verify

with open("supporting_facts_generate_dataset_full_2300_2700.json",'w') as f:
    json.dump(supporting_facts_dataset,f)