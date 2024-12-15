LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers -2 -1 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_llama8b_PopQA" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_popqa.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"