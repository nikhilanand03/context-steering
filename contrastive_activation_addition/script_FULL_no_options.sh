####### NOTE ---> Remember to only uncomment one step at a time (steps 1 and 3 are for Adobe to run)


############## STEP 1: generate_vectors #############

rm -r analysis
rm -r results
# rm -r normalized_vectors
# rm -r vectors
rm -r activations

# python generate_vectors.py --layers $(seq 0 31) --save_activations --use_latest --behaviors "context-focus" --override_dataset "generate_dataset_no_options.json" --no_options
# python normalize_vectors.py --use_latest

# Here, plot_activations.py will plot the activations of the positive and negative classes (no A/B distinction)
# python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "context-focus" --no_options

# I create a dataset called "test_dataset_oe_best_layer_no_options.json"
# This uses same examples as test_dataset_ab_multi_context_2500.json except has no options!!
python prompting_with_steering.py --layers $(seq 0 31) --use_latest --multipliers -2 -1 0 1 2 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_best_layer_no_options.json" --no_options

############## STEP 2: we run this locally and then get the best layer using the scores ##############

# Note: Custom paths dir leads to the scorable folder - make sure to save scorable versions into that.
# python scoring.py --behaviors "context-focus" --custom_path_dir "results/context-focus/scorable"
# python average_scores.py --custom_path_dir "results/context-focus/scorable"

############# STEP 3: we set the best layer and uncomment the next lines for Anirudh to run. ############

# BESTLAYER=12
# python prompting_with_steering.py --layers $BESTLAYER --use_latest --multipliers -2 -1 0 1 2 3 4 5 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=hotpotqa_multicontext_failures_625.json" --no_options

############### STEP 4: run this locally to score everything ##############

# Run these locally, rather than on the GPU
# python scoring.py --behaviors "context-focus" --multicontext
# python average_scores.py

############################