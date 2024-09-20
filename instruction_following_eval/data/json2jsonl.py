import json
import sys

def convert_json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as f:
        for item in data:
            jsonl_item = {
                "prompt": item["question"],
                "response": item["model_output"]
            }
            f.write(json.dumps(jsonl_item) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json2jsonl.py input.json output.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_json_to_jsonl(input_file, output_file)
    print(f"Conversion complete. Output written to {output_file}")