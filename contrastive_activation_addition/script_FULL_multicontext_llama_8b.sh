# For someone else to run this script as a whole, only normalised vectors is transferred through github
# So, the remaining folders (analysis and results) need to be created first before running.

############################### 1 ####################################

# rm -r analysis
# rm -r results
# rm -r normalized_vectors
# rm -r vectors

## < ---------------------------------------------------------------->

# Train vecs using "generate_dataset_long_contexts_dummy.json"
# (I changed generate_vectors to allow multicontext and tokenize differently in the case of multicontext)
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_multicontext_2500.json" --multicontext
python normalize_vectors.py --use_latest
python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_multicontext_2500.json"

# TESTING ON MCQ DATASET TO GET BEST LAYER 
# (I changed prompt with steering to account for multicontext in the type="ab" case here)
python prompting_with_steering.py --layers $(seq 0 31) --use_latest --multipliers -1 0 1 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_multicontext_2500.json" --multicontext
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "context-focus"

############################### 1 ####################################

# First run (1), commenting (2) out. Then run (2) commenting (1) out. 
# Don't delete any folders after (1) is done running. Just continue the run with (2).
# It'll use the existing outputs and continue. Only rm -r when you're starting a new run
# (doing rm -r only when anirudh has shared the existing folders)

############################### 2 ####################################

BESTLAYER=14 ## GET BEST LAYER HERE

# MULTIPLE MULTIPLIER TESTING ON MCQ DATASET

python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 0 1 2 3 4 5 6 --type ab --behaviors "context-focus" --override_ab_dataset "test_dataset_ab_multicontext_2500.json" --multicontext
python plot_results.py --layers $BESTLAYER --multipliers 0 1 2 3 4 5 6 --type ab --behaviors "context-focus"

# OPEN-ENDED TESTING (multicontext hotpotQA dataset)

python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 0 1 2 3 4 5 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=hotpotqa_multicontext.json" --multicontext
# python scoring.py --behaviors "context-focus" --multicontext
# python average_scores.py

# OPEN-ENDED TESTING (FAILURES DS)

python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers 0 1 2 3 4 5 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=hotpotqa_multicontext_failures_625.json" --multicontext
# python scoring.py --behaviors "context-focus" --multicontext
# python average_scores.py

############################### 2 ####################################