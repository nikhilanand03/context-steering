## < ------------------------------------------------------------------ > 
## BEFORE RUNNING, CHECK THAT helpers.py CONTAINS THE MISTRAL MODEL WE WANT.
## Before running scoring.py, run the convert750cq2scorable.ipynb to extract the short answers.
## < ------------------------------------------------------------------ > 
## < ------------------------------------------------------------------ > 
## STEP 1 - TEST LONG ANSWERS

# Train vecs using "generate_dataset_mistral_ans_b4_opt.json"
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_mistral --behaviors "context-focus" --override_dataset "generate_dataset_mistral_ans_b4_opt.json"
python normalize_vectors.py --use_mistral
python plot_activations.py --layers $(seq 0 31) --use_mistral --behaviors "context-focus" --override_dataset "generate_dataset_mistral_ans_b4_opt.json"

# Get best layer with "test_dataset_ab_mistral_ans_b4_opt.json"
python prompting_with_steering.py --layers $(seq 0 31) --use_mistral --multipliers -1 0 1 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_mistral_ans_b4_opt.json"
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "context-focus"

BESTLAYER=15 ## GET BEST LAYER HERE

python prompting_with_steering.py --layers $BESTLAYER --use_mistral --multipliers -3 -2 -1 0 1 2 3 4 5 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_mistral_ans_b4_opt.json"
python plot_results.py --layers $BESTLAYER --multipliers -3 -2 -1 0 1 2 3 4 5 --type ab --behaviors "context-focus"

# Test using "test_dataset_open_ended_gpt_cleaning_750_cqformat.json"
python prompting_with_steering.py --layers $BESTLAYER --use_mistral --multipliers 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json"

# Don't forget to convert2scorable.ipynb

python scoring.py --behaviors "context-focus"
python average_scores.py