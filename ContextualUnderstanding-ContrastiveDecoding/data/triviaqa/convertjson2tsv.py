import json
import csv

json_file = 'triviaqa_val_final.json'  # Replace with your JSON file name
tsv_file = 'triviaqa_test.tsv'    # Replace with your desired TSV output file name

with open(json_file, 'r') as f:
    data = json.load(f)

if isinstance(data, list):
    with open(tsv_file, 'w', newline='', encoding='utf-8') as tsv_out:
        # Get field names from the first dictionary
        fieldnames = data[0].keys()

        # Create a CSV writer with tab delimiter
        writer = csv.DictWriter(tsv_out, fieldnames=fieldnames, delimiter='\t')

        # Write the header
        writer.writeheader()

        # Write the rows
        writer.writerows(data)

    print(f"JSON data successfully written to {tsv_file}")
else:
    print("JSON data is not in the expected format (list of dictionaries).")
