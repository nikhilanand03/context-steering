# NQ dataset, Mistral 7B Few-shot prompting

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_NQ" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_nq.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

############################################

# TriviaQA dataset, Mistral 7B Few-shot prompting

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_TQA" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_triviaqa.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

############################################

# PopQA dataset with BM25 retriever, Mistral 7B Few-shot prompting

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 1 2 3 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_PQA" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_popqa_bm25.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

############################################