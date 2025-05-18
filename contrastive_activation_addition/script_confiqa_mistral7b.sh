# paths: 
#   - datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-QA.json
#   - datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MR.json
#   - datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MC.json

# mistral-7b

#################

# QA1

LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers 1 2 3 \
    --type open_ended \
    --use_mistral \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_mistral7b_confiqa_QA" \
    --sample 100 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-QA.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

python evaluate_confiqa.py \
    --input_path "results_mistral7b_confiqa_QA/results_layer=12_multiplier=1.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-QA.json" \
    --log_path "results_mistral7b_confiqa_QA/final_logs_m=1_QA"

python evaluate_confiqa.py \
    --input_path "results_mistral7b_confiqa_QA/results_layer=12_multiplier=2.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-QA.json" \
    --log_path "results_mistral7b_confiqa_QA/final_logs_m=2_QA"

python evaluate_confiqa.py \
    --input_path "results_mistral7b_confiqa_QA/results_layer=12_multiplier=3.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-QA.json" \
    --log_path "results_mistral7b_confiqa_QA/final_logs_m=3_QA"

#####################

# QA2

# LAYER=12
# BESTMULT=3

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_latest \
#     --multipliers $BESTMULT \
#     --type open_ended \
#     --use_mistral \
#     --confiqa \
#     --behaviors "context-focus" \
#     --suffix "_mistral7b_confiqa_QA_1000" \
#     --sample 1000 \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-QA.json" \
#     --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_QA/results_layer=12_multiplier=1.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-QA.json" \
#     --log_path "results_mistral7b_confiqa_QA/final_logs_m=1_QA"


#####################

# MR1

LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers 1 2 3 \
    --type open_ended \
    --use_mistral \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_mistral7b_confiqa_MR" \
    --sample 150 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MR.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_MR/results_layer=12_multiplier=1.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-MR.json" \
#     --log_path "results_mistral7b_confiqa_MR/final_logs_m=1_MR"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_MR/results_layer=12_multiplier=2.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-MR.json" \
#     --log_path "results_mistral7b_confiqa_MR/final_logs_m=2_MR"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_MR/results_layer=12_multiplier=3.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-MR.json" \
#     --log_path "results_mistral7b_confiqa_MR/final_logs_m=3_MR"

#####################

# MR2

# LAYER=12
# BESTMULT=2

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_latest \
#     --multipliers $BESTMULT \
#     --type open_ended \
#     --use_mistral \
#     --confiqa \
#     --behaviors "context-focus" \
#     --suffix "_llama8b_confiqa_MR_1000" \
#     --sample 1000 \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MR.json" \
#     --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

# python evaluate_confiqa.py \
#     --input_path "results_llama8b_confiqa_MR/results_layer=12_multiplier=1.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-MR.json" \
#     --log_path "results_llama8b_confiqa_MR/final_logs_m=1_MR"

#######################

# MC1

LAYER=12

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers 1 2 3 \
    --type open_ended \
    --use_mistral \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_mistral7b_confiqa_MC" \
    --sample 150 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MC.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_MC/results_layer=12_multiplier=1.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-MC.json" \
#     --log_path "results_mistral7b_confiqa_MC/final_logs_m=1_MC"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_MC/results_layer=12_multiplier=2.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-MC.json" \
#     --log_path "results_mistral7b_confiqa_MC/final_logs_m=2_MC"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_MC/results_layer=12_multiplier=3.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-MC.json" \
#     --log_path "results_mistral7b_confiqa_MC/final_logs_m=3_MC"

#####################

# MC2

# LAYER=12
# BESTMULT=2

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_latest \
#     --multipliers $BESTMULT \
#     --type open_ended \
#     --use_mistral \
#     --confiqa \
#     --behaviors "context-focus" \
#     --suffix "_mistral7b_confiqa_MC_1000" \
#     --sample 1000 \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MC.json" \
#     --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

# python evaluate_confiqa.py \
#     --input_path "results_mistral7b_confiqa_QA/results_layer=12_multiplier=1.0_behavior=context-focus_type=open_ended_use_base_model=False_model_size=7b_dataset_path=test_dataset_oe_ConFiQA-QA.json" \
#     --log_path "results_mistral7b_confiqa_QA/final_logs_m=1_QA"

################################