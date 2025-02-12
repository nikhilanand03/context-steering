import csv
import sys

csv.field_size_limit(sys.maxsize)

def replace_internal_tabs(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        for row in reader:
            cleaned_row = [field.replace('\t', ' ') for field in row]
            writer.writerow(cleaned_row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.tsv output.tsv")
    else:
        replace_internal_tabs(sys.argv[1], sys.argv[2])