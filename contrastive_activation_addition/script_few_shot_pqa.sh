# PopQA dataset with Contriever retriever, Mistral 7B Few-shot prompting
# Make sure MISTRAL_LIKE_MODEL is set in helpers.py

LAYER=13

python prompting_with_steering.py \
    --layers $LAYER \
    --use_mistral \
    --multipliers 0 2 \
    --type open_ended \
    --behaviors "context-focus" \
    --suffix "_mistral7b_PQA_Contriever_FS" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_new_popqa_contriever.json" \
    --few_shot \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt" \
    --max_char_limit 100000

############################################