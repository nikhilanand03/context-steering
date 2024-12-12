import json

# Read the JSONL file
input_file = 'train.jsonl'
output_file = 'train.json'

with open(input_file, 'r') as infile:
    # Read lines and parse each line as a JSON object
    json_objects = [json.loads(line) for line in infile]

# Write the JSON objects to a JSON file
with open(output_file, 'w') as outfile:
    json.dump(json_objects, outfile, indent=4)

print(f"Converted {input_file} to {output_file}")