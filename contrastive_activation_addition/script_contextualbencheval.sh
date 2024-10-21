# We need to fetch the vector that we're interested in while prompting.
# VOI : ./1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt

# MULTIHOP-QA
python prompting_with_steering.py --layers 12 --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=multihopqa.json"
python scoring.py --behaviors "context-focus"
python average_scores.py

# REMAINING DATASETS
# HOTPOT-QA
python prompting_with_steering.py --layers 12 --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=hotpotqa.json"
# TRIVIA-QA
python prompting_with_steering.py --layers 12 --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=triviaqa.json"
# TRUTHFUL-QA
python prompting_with_steering.py --layers 12 --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=truthfulqa.json"

python scoring.py --behaviors "context-focus"
python average_scores.py

# TESTING RAG_MULTIHOP_QA WITH VANILLA LLAMA-3.1-8B TO COMPARE RESULTS WITH PAPER
# ASSUMING STANDARD RAG FORMAT IS FOLLOWED AND WILL EVALUATE USING EXACT-MATCH
# RAG-MULTIHOP-QA
python prompting_with_steering.py --layers 12 --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" --use_latest --multipliers 0 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=ragmultihopqa.json"

# TESTING MUSIQUE DATASET WITH THE SUPPORTING CONTEXTS (SHORT CONTEXTS)
# TO VERIFY THAT LONG CONTEXTS IS THE REASON STEERING IS FAILING
python prompting_with_steering.py --layers 12 --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=musique.json"
python scoring.py --behaviors "context-focus"
python average_scores.py