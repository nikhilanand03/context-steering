mkdir analysis/context-focus
mkdir results
# For someone else to run this script as a whole, only normalised vectors is transferred through github
# So, the remaining folders (analysis and results) need to be created first before running.

## < ---------------------------------------------------------------->

# Train vecs using "generate_dataset_long_contexts_dummy.json"
# (I changed generate_vectors to allow multicontext and tokenize differently in the case of multicontext)
# python generate_vectors.py --layers $(seq 0 31) --save_activations --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_multicontext_2500.json" --multicontext
# python normalize_vectors.py --use_latest
# python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_multicontext_2500.json"

# TESTING ON MCQ DATASET TO GET BEST LAYER 
# (I changed prompt with steering to account for multicontext in the type="ab" case here)
python prompting_with_steering.py --layers $(seq 0 31) --use_latest --multipliers -1 0 1 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_multicontext_2500.json" --multicontext
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "context-focus"

BESTLAYER=12 ## GET BEST LAYER HERE

# MULTIPLE MULTIPLIER TESTING ON MCQ DATASET

python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_multicontext_2500.json" --multicontext
python plot_results.py --layers $BESTLAYER --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus"

# OPEN-ENDED TESTING (multicontext hotpotQA dataset)

python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=hotpotqa_multicontext.json" --multicontext
python scoring.py --behaviors "context-focus"; python average_scores.py
