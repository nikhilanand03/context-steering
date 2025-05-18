
python generate_vectors.py \
    --layers $(seq 10 14) \
    --save_activations \
    --use_latest \
    --behaviors "context-focus" \
    --override_dataset "generate_dataset_contrastive.json" \
    --contrastive

python normalize_vectors.py --use_latest

python plot_activations.py --layers $(seq 10 14) --use_latest --behaviors "context-focus" --no_options

######################

# Keep changing BEST LAYER and get all plots
BESTLAYER=12

python prompting_with_steering.py \
    --layers $BESTLAYER \
    --use_latest \
    --multipliers 2 \
    --type ab \
    --behaviors "context-focus" \
    --override_ab_dataset "test_dataset_ab_failures_llama_induce_output.json" \

python plot_results.py --layers 12 --multipliers 2 --type ab --behaviors "context-focus"

#####################

BESTLAYER=11

python prompting_with_steering.py \
    --layers $BESTLAYER \
    --use_latest \
    --multipliers 2 \
    --type open_ended \
    --behaviors "context-focus" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json" \
    --override_vector_path "normalized_vectors/context-focus/vec_layer_${BESTLAYER}_Meta-Llama-3.1-8B-Instruct.pt"

