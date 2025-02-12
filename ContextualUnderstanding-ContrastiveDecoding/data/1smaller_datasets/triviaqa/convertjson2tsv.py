import json
import csv

json_file = 'triviaqa_FINAL.json'
tsv_file = 'triviaqa_test.tsv'

with open(json_file, 'r') as f:
    data = json.load(f)

if isinstance(data, list):
    with open(tsv_file, 'w', newline='', encoding='utf-8') as tsv_out:
        fieldnames = data[0].keys()

        writer = csv.DictWriter(tsv_out, fieldnames=fieldnames, delimiter='\t')

        writer.writeheader()

        writer.writerows(data)

    print(f"JSON data successfully written to {tsv_file}")
else:
    print("JSON data is not in the expected format (list of dictionaries).")
