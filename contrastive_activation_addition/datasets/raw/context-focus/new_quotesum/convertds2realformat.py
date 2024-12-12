import json

def convert_json(data):
    converted = {
        "question": data["question"],
        "answer": [
            data["short_ans_1"],
            data["short_ans_2"],
            data["short_ans_3"],
            data["short_ans_4"],
            data["short_ans_5"],
            data["short_ans_6"],
            data["short_ans_7"],
            data["short_ans_8"]
        ],
        "rag": [
            data["source1"],
            data["source2"],
            data["source3"],
            data["source4"],
            data["source5"],
            data["source6"],
            data["source7"],
            data["source8"]
        ]
    }

    converted["answer"] = [ans for ans in converted["answer"] if ans]
    converted["rag"] = [rag for rag in converted["rag"] if rag]

    return converted

with open("test.json",'r') as f:
    input_data = json.load(f)

output_data = []

for item in input_data:
    output_item = convert_json(item)
    output_data.append(output_item)

with open("converted_test.json",'w') as f:
    json.dump(output_data,f)