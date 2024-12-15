import csv
import json

def parse_value(value):
    # Try to parse the value as a number
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            pass

    # Try to parse the value as a list
    if value.startswith('[') and value.endswith(']'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # Return as a string if no parsing works
    return value

# Function to convert TSV to JSON
def convert_tsv_to_json(tsv_file, json_file):
    try:
        # Read the TSV file and parse its content
        with open(tsv_file, 'r') as infile:
            reader = csv.DictReader(infile, delimiter='\t')
            data = []
            for row in reader:
                parsed_row = {key: parse_value(value) for key, value in row.items()}
                data.append(parsed_row)

        # Write the data to a JSON file
        with open(json_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        print(f"Successfully converted {tsv_file} to {json_file}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    tsv_file = "popQA.tsv"  # Replace with your TSV file name
    json_file = "popQA.json"  # Replace with your desired JSON file name
    convert_tsv_to_json(tsv_file, json_file)
