LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 1 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQ_FS_M=13" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_nq.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --sample 1000 \
    --max_fs_length 15000

#############################

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 1 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_TQA_FS_M=13" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_triviaqa.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --sample 1000 \
    --max_fs_length 15000

#############################

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 1 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_PQA_Contr_FS_M=13" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_popqa_contriever.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --sample 1000 \
    --max_fs_length 15000

#############################

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 1 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQswap_FS_M=13" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_nq_swap.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --sample 1000 \
    --max_fs_length 15000

#############################