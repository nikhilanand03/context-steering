## JUST CHANGE THE MISTRAL_LIKE_MODEL CONSTANT IN HELPERS.PY
## AND WHEREVER YOU SEE CONDITIONS (IN BASELINE.PY, CONTR+STEER.PY, LLAMA_WRAPPER.PY, TOKENIZE.PY) - 
## WRITE AN ADDITIONAL CONDITION FOR YOUR MODEL OF INTEREST
## < ------------------------------------------------------------------ > 
## Remember that this is to be run SEPARATELY from any other pipeline. We don't have a means to separate ablation results from non-ablation ones for now.
## < ------------------------------------------------------------------ > 
## STEP 1 - TEST LONG ANSWERS

# Train vecs using "generate_llama_induce_output.json"
python generate_vectors.py --layers $(seq 0 17) --save_activations --use_mistral --behaviors "context-focus" --override_dataset "generate_dataset_llama_induce_output_gemma.json"
python normalize_vectors.py --use_mistral
python plot_activations.py --layers $(seq 0 17) --use_mistral --behaviors "context-focus" --override_dataset "generate_dataset_llama_induce_output_gemma.json"

## Get best layer with "test_ab_failures_llama_induce_output.json"
# Ablate the vector and get results. These are stored as mult=1
python prompting_with_steering.py --layers $(seq 0 17) --use_mistral --multipliers 1 --ablate --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_failures_llama_induce_output_gemma.json"
# Without any steering get results. These are stored as mult=0
python prompting_with_steering.py --layers $(seq 0 17) --use_mistral --multipliers 0 --ty`pe ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_failures_llama_induce_output_gemma.json"
# Plot the results
python plot_results.py --layers $(seq 0 17) --multipliers 0 1 --type ab --behaviors "context-focus"

BESTLAYER=9 ## GET BEST LAYER HERE

# Test using "test_oe_failures_fixed_geval_87%.json"
# python prompting_with_steering.py --layers $BESTLAYER --use_mistral --multipliers 0 1 1.2 1.4 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_failures_fixed_geval_87%.json"

# Test using "test_oe_successes.json"
python prompting_with_steering.py --layers $BESTLAYER --use_mistral --ablate --multipliers 1 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_successes.json"
python prompting_with_steering.py --layers $BESTLAYER --use_mistral --multipliers 0 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_successes.json"

# Test using "test_dataset_open_ended_gpt_cleaning_750_cqformat.json"
# python prompting_with_steering.py --layers $BESTLAYER --use_mistral --multipliers 0 1 1.2 1.4 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json"

# python scoring.py --behaviors "context-focus"
# python average_scores.py