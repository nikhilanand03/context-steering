###### MISTRAL-7B ##########################################

# mistral-7b
# GO TO helpers.py AND MAKE SURE THE MISTRAL_LIKE_MODEL IS SET TO "MISTRAL-7B"

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 4 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQ_sysprompt" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_nq.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 4 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_PopQA_sysprompt" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_popqa.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 4 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_TriviaQA_sysprompt" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_triviaqa.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"


# python scoring.py --behaviors "context-focus"; python average_scores.py

##############################################

############ LLAMA-70B  ############
# (Need to change mistral-like-model before running this)
# GO TO helpers.py AND MAKE SURE THE MISTRAL_LIKE_MODEL IS SET TO "LLAMA-70B"

# LAYER=32

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_mistral \
#     --system_prompt "pos" \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama70b_NQ_sysprompt" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_nq.json" \
#     --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_mistral \
#     --system_prompt "pos" \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama70b_PopQA_sysprompt" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_popqa.json" \
#     --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_mistral \
#     --system_prompt "pos" \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama70b_TriviaQA_sysprompt" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_triviaqa.json" \
#     --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"


# python scoring.py --behaviors "context-focus"; python average_scores.py

##############################################