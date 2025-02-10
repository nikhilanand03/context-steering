# TESTING (DEBUGGING)

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQ_FS_debug" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_nq.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --debug

############################################

# NQ dataset, Mistral 7B Few-shot prompting
# Make sure MISTRAL_LIKE_MODEL is set in helpers.py

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQ_FS" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_nq.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --max_char_limit 100000

############################################

# TriviaQA dataset, Mistral 7B Few-shot prompting
# Make sure MISTRAL_LIKE_MODEL is set in helpers.py

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_TQA_FS" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_triviaqa.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --max_char_limit 100000


############################################

# PopQA dataset with BM25 retriever, Mistral 7B Few-shot prompting
# Make sure MISTRAL_LIKE_MODEL is set in helpers.py

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_PQA_BM25_FS" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_popqa_bm25.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --max_char_limit 100000

############################################