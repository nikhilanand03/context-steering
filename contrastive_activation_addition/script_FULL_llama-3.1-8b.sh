## < ------------------------------------------------------------------ > 
## STEP 1 - TEST SHORT ANSWERS

# Train vectors using "generate_dataset_short_ans_cqformat.json"
python generate_vectors.py --layers $(seq 0 31) --save_activations --suffix "_short_ans" --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_short_ans_cqformat.json"
python normalize_vectors.py --use_latest --suffix "_short_ans"
python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "context-focus" --suffix "_short_ans" --override_dataset "generate_dataset_short_ans_cqformat.json"

# Get best layer with "test_dataset_ab_short_ans_cqformat.json"
python prompting_with_steering.py --layers $(seq 0 31) --suffix "_short_ans" --use_latest --multipliers -1 0 1 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_short_ans_cqformat.json"
python plot_results.py --layers $(seq 0 31) --suffix "_short_ans" --multipliers -1 1 --type ab --behaviors "context-focus"

BESTLAYER=13 ## GET BEST LAYER HERE

python prompting_with_steering.py --layers $BESTLAYER --suffix "_short_ans" --use_latest --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_short_ans_cqformat.json"
python plot_results.py --layers $BESTLAYER --suffix "_short_ans" --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus"

# Test using "test_oe_failures_fixed_geval_87%.json"
python prompting_with_steering.py --layers $BESTLAYER --suffix "_short_ans" --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_failures_fixed_geval_87%.json"

# Test using "test_oe_successes.json"
python prompting_with_steering.py --layers $BESTLAYER --suffix "_short_ans" --use_latest --multipliers 0 -1 -2 -3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_successes.json"

# Test using "test_dataset_open_ended_1200.json"
# python prompting_with_steering.py --layers $BESTLAYER --suffix "_short_ans" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_1200.json"

# Test using "test_dataset_open_ended_gpt_cleaning_750_cqformat.json"
python prompting_with_steering.py --layers $BESTLAYER --suffix "_short_ans" --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json"

python scoring.py --behaviors "context-focus" --suffix "_short_ans"
python average_scores.py --suffix "_short_ans"

## < ------------------------------------------------------------------ > 
## STEP 2 - TEST LONG ANSWERS

# Train vecs using "generate_llama_induce_output.json"
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_llama_induce_output.json"
python normalize_vectors.py --use_latest
python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_llama_induce_output.json"

# Get best layer with "test_ab_failures_llama_induce_output.json"
python prompting_with_steering.py --layers $(seq 0 31) --use_latest --multipliers -1 0 1 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_failures_llama_induce_output.json"
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "context-focus"

BESTLAYER=12 ## GET BEST LAYER HERE

python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_failures_llama_induce_output.json"
python plot_results.py --layers $BESTLAYER --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus"

# Test using "test_oe_failures_fixed_geval_87%.json"
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_failures_fixed_geval_87%.json"

# Test using "test_oe_successes.json"
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -1 -2 -3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_successes.json"

# Test using "test_dataset_open_ended_1200.json"
# python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_1200.json"

# Test using "test_dataset_open_ended_gpt_cleaning_750_cqformat.json"
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json"

python scoring.py --behaviors "context-focus"; python average_scores.py

## < ------------------------------------------------------------------ > 
## STEP 3 - TEST BASELINE METHODS

# Run the baseline methods on the geval_87% dataset saving to same results folder
# python baseline_methods.py

# Run the baseline methods on the 750-element dataset saving to same results folder. Note that without --usemistral, we stick to default (Llama-8B)
python baseline_methods.py --long

# Score all the newly generated results files (with *open_ended* in it)
python scoring.py --behaviors "context-focus"

# This will compute all average scores in the full results/open_ended_scores directory
python average_scores.py

## < ------------------------------------------------------------------ > 
## STEP 4 - CONTRASTIVE STEERING

# Will create two new sets of results files
python contrastive+steering.py --layer $BESTLAYER --multipliers 1 2 3 --alpha 0.5 --thresh 0.0
python contrastive+steering.py --layer $BESTLAYER --multipliers 1 2 3 --alpha 4.0 --thresh 0.001

# Will score everything and save it
python scoring.py --behaviors "context-focus"; python average_scores.py

# Will plot only contrastive results and save with a "method" suffix
# python plot_results.py --layers $BESTLAYER --multipliers 1 2 3 --type open_ended --method "contrastive-steer" --behaviors "context-focus" --title "Layer 12 - Llama 3.1 8B Instruct"

## < ------------------------------------------------------------------ > 
## STEP 5 - IF EVAL

# PARALLEL 1,2,3,4 ->
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 0 --type if_eval --behaviors "context-focus"
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 2 --type if_eval --behaviors "context-focus"
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 1 --type if_eval --behaviors "context-focus"
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 3 --type if_eval --behaviors "context-focus"
python baseline_methods.py --type "if_eval"

## < ------------------------------------------------------------------ > 
## STEP 6 - OTHER DATASETS

# Testing same vectors on irrelevant dataset
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -3 -2 -1 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=irrel.json"

# Testing same vectors on mildly relevant dataset
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -3 -2 -1 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=mild_rel.json"

# Testing same vectors on long context dataset
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -3 -2 -1 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=longcontext_num_contexts=20.json"

# Testing same vectors on copyright dataset
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -3 -2 -1 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=copyright.json"

# Testing same vectors on long context failures dataset
python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -3 -2 -1 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_failures_version=longcontext_num_contexts=20.json"

# Will score everything and save it
python scoring.py --behaviors "context-focus"; python average_scores.py