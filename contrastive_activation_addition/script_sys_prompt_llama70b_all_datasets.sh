############ LLAMA-70B  ############
# (Need to change mistral-like-model before running this)
# GO TO helpers.py AND MAKE SURE THE MISTRAL_LIKE_MODEL IS SET TO "LLAMA-70B"

LAYER=32

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_llama70b_NQ_sysprompt" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_nq.json" \
    --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_llama70b_PopQA_sysprompt" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_popqa.json" \
    --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_llama70b_TriviaQA_sysprompt" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_triviaqa.json" \
    --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

# python scoring.py --behaviors "context-focus"; python average_scores.py

##############################################