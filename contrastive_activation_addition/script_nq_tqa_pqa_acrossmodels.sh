##############################################
###### LLAMA-3.1-8B ##########################

# LAYER=12

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_latest \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama8b_NQ" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_nq.json" \
#     --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_latest \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama8b_PopQA" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_popqa.json" \
#     --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_latest \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama8b_TriviaQA" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_triviaqa.json" \
#     --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

##############################################
##############################################

###### MISTRAL-7B ######################################################################
###### in helpers.py put MISTRAL_LIKE_MODEL as Mistral-7B before running the following ####

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQ" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_nq.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_PopQA" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_popqa.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_TriviaQA" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_triviaqa.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

##############################################
##############################################

###### LLAMA-70B ######################################################################
###### in helpers.py put MISTRAL_LIKE_MODEL as Llama-70B before running the following ####

# LAYER=32

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_mistral \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama70b_NQ" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_nq.json" \
#     --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_mistral \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama70b_PopQA" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_popqa.json" \
#     --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_mistral \
#     --multipliers -2 -1 0 1 2 3 \
#     --type open_ended \
#     --behaviors "context-focus" \
#     --suffix "_llama70b_TriviaQA" \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_triviaqa.json" \
#     --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

##############################################
##############################################