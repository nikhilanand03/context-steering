import json

with open("final_generate_dataset.json", 'r') as f:
    data = json.load(f)

with open("hotpot_training_data.json", 'r') as f:
    data_tr = json.load(f)[:3000]

supporting_facts_dataset = []

print(len(data), len(data_tr))

i = 0
for item in data_tr:
    if i>=len(data):
        break

    if item['question'] == data[i]['question']:
        item_generate_ds = data[i]
        i += 1

        supp_facts = item['supporting_facts']
        contexts = item['context']

        matched_contexts = []
        for fact in supp_facts:
            fact_title = fact[0]
            for context in contexts:
                if context[0] == fact_title:
                    matched_contexts.append("".join(context[1]))
        
        ans_matching = item_generate_ds['answer_matching_behavior']

        new_item = {
            'question': item['question'],
            'answer': item['answer'],
            'supporting_contexts': matched_contexts,
            'model_output': item_generate_ds['options'][1 if ans_matching=='(A)' else 0]
        }

        supporting_facts_dataset.append(new_item)

print(supporting_facts_dataset[0])  # Print the first item to verify

with open("supporting_facts_generate_dataset.json",'w') as f:
    json.dump(supporting_facts_dataset,f)