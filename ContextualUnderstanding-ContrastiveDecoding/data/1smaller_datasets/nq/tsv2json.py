#!/usr/bin/env python3
import sys
import json
import csv
from typing import List, Dict
from pathlib import Path

def tsv_to_json(input_file: str, output_file: str = None) -> List[Dict]:
    """
    Convert a TSV file to JSON format.
    
    Args:
        input_file (str): Path to the input TSV file
        output_file (str, optional): Path to save the JSON output. If None, prints to stdout
        
    Returns:
        List[Dict]: List of dictionaries containing the TSV data
    """
    try:
        # Read TSV file
        with open(input_file, 'r', encoding='utf-8') as tsv_file:
            # Use csv.DictReader with tab delimiter
            reader = csv.DictReader(tsv_file, delimiter='\t')
            # Convert to list of dictionaries
            data = [row for row in reader]
        
        # Write to file or stdout
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
            print(f"Successfully converted {input_file} to {output_file}")
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
        return data
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python tsv2json.py input.tsv [output.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    tsv_to_json(input_file, output_file)

if __name__ == "__main__":
    main()