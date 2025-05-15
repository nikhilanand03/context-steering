# paths: 
#   - datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-QA.json
#   - datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MR.json
#   - datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MC.json

# llama-3.1-8b

#################

# QA

LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers 1 2 3 \
    --type open_ended \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_llama8b_confiqa_QA" \
    --sample 250 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-QA.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

python scoring.py --behaviors "context-focus"; python average_scores.py

#####################

# MR

LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers 1 2 3 \
    --type open_ended \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_llama8b_confiqa_MR" \
    --sample 250 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MR.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

#######################

# MC

LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers 1 2 3 \
    --type open_ended \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_llama8b_confiqa_MC" \
    --sample 250 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MC.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

################################