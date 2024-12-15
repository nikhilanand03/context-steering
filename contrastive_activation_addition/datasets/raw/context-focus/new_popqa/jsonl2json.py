import json

# Function to convert JSONL to JSON
def convert_jsonl_to_json(jsonl_file, json_file):
    try:
        # Open the JSONL file and read all lines
        with open(jsonl_file, 'r') as infile:
            data = [json.loads(line) for line in infile]

        # Write the data to a JSON file
        with open(json_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        print(f"Successfully converted {jsonl_file} to {json_file}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    jsonl_file = "popqa_retrieved_docs.jsonl"  # Replace with your JSONL file name
    json_file = "popqa_retrieved_docs.json"   # Replace with your desired JSON file name
    convert_jsonl_to_json(jsonl_file, json_file)
