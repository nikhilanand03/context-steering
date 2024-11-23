# Here, we'll generate vectors using the "with" and "without" context prompts.
# Then we'll have to evaluate open-ended generations on a held-out test set in order to determine the best vector.

# ----------------------------------------------------------------------------------------- #

# We MUST override dataset if running --no_options, since generate_dataset_no_options.json has a specific format that is necessary.
# Here, plot_activations.py will plot the activations of the positive and negative classes (no A/B distinction)

python generate_vectors.py --layers $(seq 0 31) --save_activations --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_no_options.json" --no_options
python normalize_vectors.py --use_latest
python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "context-focus" --no_options

# I create a dataset called "test_dataset_oe_best_layer_no_options.json"
# This uses same examples as test_dataset_ab_multi_context_2500.json except has no options!!
python prompting_with_steering.py --layers $(seq 0 31) --use_latest --multipliers -2 -1 0 1 2 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_best_layer_no_options.json" --no_options
python scoring.py --behaviors "context-focus" --multicontext
python average_scores.py

## IMPORTANT: After this, look at the accuracies and manually create a plot to get the best layer

BESTLAYER=12

python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -2 -1 0 1 2 3 4 5 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=hotpotqa_multicontext_failures_625.json" --no_options
python scoring.py --behaviors "context-focus" --multicontext
python average_scores.py