# We need to fetch the vector that we're interested in while prompting.
# VOI : ./1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt

# MULTIHOP-QA
python prompting_with_steering.py --layers 12 --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=multihopqa.json"
python scoring.py --behaviors "context-focus"
python average_scores.py