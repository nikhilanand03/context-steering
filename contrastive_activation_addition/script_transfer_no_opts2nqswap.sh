# We'll do the same as script_FULL_llama-3.1-8b.sh
# The only difference is using the no_options vector while steering.

###############  PART 1   ###############
# (First run this, to get the best layer)

rm -r analysis
rm -r results
rm -r normalized_vectors
rm -r vectors
rm -r activations

# Get best layer with "test_ab_failures_llama_induce_output.json"
for NUM in {0..31}; do
  python prompting_with_steering.py \
    --layers $NUM \
    --use_latest \
    --multipliers -1 0 1 \
    --type ab \
    --behaviors "context-focus" \
    --override_ab_dataset "test_dataset_ab_failures_llama_induce_output.json" \
    --override_vector_path "1COMPLETED_RUNS/39_llama-3.1-8b_no-options_max_tokens/normalized_vectors/context-focus/vec_layer_${NUM}_Meta-Llama-3.1-8B-Instruct.pt"
done

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "context-focus"

###########################################

###############  PART 2    ###############
# (Next run this, to set the best layer and get results)

# Determine the best layer from the plot.
# BESTLAYER=?

# python prompting_with_steering.py \
#     --layers $BESTLAYER \
#     --use_latest \
#     --multipliers -3 -2 -1 0 1 2 3 \
#     --type ab \
#     --behaviors "context-focus" \
#     --override_ab_dataset "test_dataset_ab_failures_llama_induce_output.json" \
#     --override_vector_path "1COMPLETED_RUNS/39_llama-3.1-8b_no-options_max_tokens/normalized_vectors/context-focus/vec_layer_${BESTLAYER}_Meta-Llama-3.1-8B-Instruct.pt"

# python plot_results.py --layers $BESTLAYER --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus"

# python prompting_with_steering.py \
#     --layers $BESTLAYER \
#     --use_latest \
#     --multipliers 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json" \
#     --override_vector_path "1COMPLETED_RUNS/39_llama-3.1-8b_no-options_max_tokens/normalized_vectors/context-focus/vec_layer_${BESTLAYER}_Meta-Llama-3.1-8B-Instruct.pt"

###########################################

# Run the below locally once all the results have been fetched.
# python scoring.py --behaviors "context-focus"; python average_scores.py