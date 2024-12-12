# SYSTEM PROMPT ACROSS MODELS

# llama-3.1-8b

rm -r analysis
rm -r results
rm -r normalized_vectors
rm -r vectors
rm -r activations

LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_llama8b" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

# python scoring.py --behaviors "context-focus"; python average_scores.py

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
    --suffix "_mistral7b" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

# python scoring.py --behaviors "context-focus"; python average_scores.py

# llama-70b
# GO TO helpers.py AND MAKE SURE THE MISTRAL_LIKE_MODEL IS SET TO "LLAMA-70B"

LAYER=32

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --system_prompt "pos" \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_llama70b" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json" \
    --override_vector_path "1COMPLETED_RUNS/30FULL_llama-70b-without-quantising/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-70B-Instruct.pt"

# python scoring.py --behaviors "context-focus"; python average_scores.py