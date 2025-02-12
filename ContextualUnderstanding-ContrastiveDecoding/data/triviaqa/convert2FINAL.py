import json
import re
from tqdm import tqdm

with open("squad_triviaqa_val.json", 'r') as f:
    squad_data = json.load(f)["data"]

with open("converted_triviaqa_val.json", 'r') as f:
    orig_data = json.load(f)['Data']


def extract_complete_segment(content, answer_start, window):
    """Extracts a segment around answer_start but ensures it starts and ends with complete sentences."""
    start = max(0, answer_start - window)
    end = min(len(content), answer_start + window)
    segment = content[start:end]

    # Adjust start: Find the nearest sentence start (a capital letter after a period)
    start_match = re.search(r'(?<=\.\s)([A-Z])', content[:start])
    if start_match:
        start = start_match.start()
    else:
        start = 0

    # Adjust end: Find the nearest period after answer_start
    end_match = re.search(r'\.', content[end:])
    if end_match:
        end = end + end_match.end()

    segment = content[start:end].strip().replace("\n", ". ")
    # final_output = ""

    # for segment in segments:
    #     if '.' in segment:
    #         if final_output.endswith("."):
    #             final_output += " "
    #         final_output += segment

    return segment

# print(len(squad_data))
# print(len(orig_data))


final_li = []
answers_empty_count = 0

for squad_item in tqdm(squad_data):
    if len(squad_item['paragraphs'][0]['qas'][0]["answers"]) == 0:
        answers_empty_count += 1
        continue

    final_item = {
        "question": squad_item['paragraphs'][0]['qas'][0]["question"],
        "ans": squad_item['paragraphs'][0]['qas'][0]["answers"][0]["text"]
    }

    assert len(squad_item['paragraphs'][0]['qas'][0]["id"].split("--")) == 2
    context_filepath = squad_item['paragraphs'][0]['qas'][0]["id"].split(
        "--")[-1]

    with open("contexts/"+context_filepath, "r") as file:
        content = file.read()

    answer_start = squad_item['paragraphs'][0]['qas'][0]["answers"][0]["answer_start"]
    excerpt = extract_complete_segment(content, answer_start, 240)
    final_item["gold_ctx"] = excerpt

    item_id = squad_item['paragraphs'][0]['qas'][0]["qid"]

    for orig_item in orig_data:
        if orig_item["QuestionId"] == item_id:
            final_item["answers"] = orig_item["Answer"]["NormalizedAliases"]
            break

    final_li.append(final_item)


print("answers_empty_count: ", answers_empty_count)

with open("triviaqa_val_final.json", 'w') as f:
    json.dump(final_li, f)
