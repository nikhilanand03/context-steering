# 1. Replace input_data_m0.jsonl with any of the files corresponding to different multipliers.
# 2. Transfer results to a different folder before subsequent runs to avoid losing results.

python3 -m evaluation_main \
  --input_data=./data/input_data.jsonl \
  --input_response_data=./data/input_response_data_cad_alpha=1.jsonl \
  --output_dir=./data/

exit 0