# TESTING (DEBUGGING)

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQSWAP_FS_debug" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_nqswap.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --debug

############################################

# TESTING NQ-SWAP on 

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQSWAP_FS" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_nqswap.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

############################################