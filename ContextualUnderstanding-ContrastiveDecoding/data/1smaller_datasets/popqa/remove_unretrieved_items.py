import csv
import json

json_questions = set()
with open('popqa_contriever_results.jsonl', 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        record = json.loads(line)
        json_questions.add(record['question'])

with open('popqa_test_unremoved.tsv', 'r', encoding='utf-8') as tsv_in, \
     open('popqa_test.tsv', 'w', encoding='utf-8', newline='') as tsv_out:
    
    reader = csv.DictReader(tsv_in, delimiter='\t')
    writer = csv.DictWriter(tsv_out, fieldnames=reader.fieldnames, delimiter='\t')
    
    writer.writeheader()
    for row in reader:
        if row['question'] in json_questions:
            writer.writerow(row)
