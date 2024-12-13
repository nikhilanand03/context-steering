# MUSIQUE - I'm just using this method on the MUSIQUE dataset. 
# Using the dynamic_m tag and only providing multipliers of 1
# Ideally this should ignore the multipliers of 1 and make the LlamaWrapper perform DynAA
# (which determines the multiplier on its own at each step)

python prompting_with_steering.py \
    --layers 12 \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt" \
    --use_latest \
    --multipliers 1 \
    --dynamic_m \
    --type open_ended \
    --behaviors "context-focus" \
    --override_oe_dataset_path "python prompting_with_steering.py" \
    --layers $BESTLAYER \
    --use_latest \
    -- suffix "_dynamic_m" \
    --multipliers 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_gpt_cleaning_750_cqformat.json"

# python scoring.py --behaviors "context-focus"; python average_scores.py