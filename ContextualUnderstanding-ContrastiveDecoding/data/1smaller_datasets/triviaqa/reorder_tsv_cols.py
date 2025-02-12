import pandas as pd
import argparse

def reorder_tsv(input_tsv, output_tsv, column_order):
    # Read the TSV file
    df = pd.read_csv(input_tsv, sep='\t')
    
    # Ensure all specified columns exist
    missing_cols = [col for col in column_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input TSV: {missing_cols}")
    
    # Reorder columns
    df = df[column_order]
    
    # Save to new TSV
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Reordered TSV saved to {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorder columns in a TSV file.")
    parser.add_argument("--input_tsv", help="Path to the input TSV file")
    parser.add_argument("--output_tsv", help="Path to save the reordered TSV file")
    parser.add_argument("--columns", nargs='+', help="Desired column order")
    
    args = parser.parse_args()
    
    reorder_tsv(args.input_tsv, args.output_tsv, args.columns)