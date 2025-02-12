import json

with open('kilt_triviaqa_subset_squad.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

question_dict = {}
empty_count = 0

# Process entries and track duplicates
for entry in data['data']:
    # Verify structure assumptions
    assert len(entry['paragraphs']) == 1, "Each entry should have exactly one paragraph."
    paragraph = entry['paragraphs'][0]
    assert len(paragraph['qas']) == 1, "Each paragraph should have exactly one Q&A."
    
    qa = paragraph['qas'][0]
    question = qa['question']
    current_answers = qa.get('answers', [])
    
    # Update empty count for initial tally
    if not current_answers:
        empty_count += 1

    if question not in question_dict:
        question_dict[question] = entry
    else:
        # Prioritize entries with answers
        existing_answers = question_dict[question]['paragraphs'][0]['qas'][0].get('answers', [])
        if not existing_answers and current_answers:
            question_dict[question] = entry

# Filter out any remaining entries without answers
final_data = []
removed_empty = 0

for entry in question_dict.values():
    qa = entry['paragraphs'][0]['qas'][0]
    if qa.get('answers'):
        final_data.append(entry)
    else:
        removed_empty += 1

data['data'] = final_data

print(f"Removed after deduplication: {removed_empty}")
print(f"Final dataset size: {len(final_data)}")

with open('deduplicated_kilt_triviaqa_subset_squad.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)