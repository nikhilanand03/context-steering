import json
import sys

def jsonl_to_json(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = [json.loads(line) for line in infile]
    
    for i in range(len(data)):
        d = data[i]
        data[i]['question'] = d['prompt']
        del data[i]['key']
        del data[i]['instruction_id_list']
        del data[i]['kwargs']
        del data[i]['prompt']
    
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jsonl2json.py <input_jsonl_file> <output_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    jsonl_to_json(input_file, output_file)
    print(f"Conversion complete. Output saved to {output_file}")