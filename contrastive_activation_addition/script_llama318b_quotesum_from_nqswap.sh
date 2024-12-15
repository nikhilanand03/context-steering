## Here we transfer the No-options vector to the task of semqa generation

rm -r "results_quotesum"

for NUM in {10..15}; do
  python prompting_with_steering.py \
    --layers $NUM \
    --use_latest \
    --multipliers -3 -2 -1 0 1 2 3 \
    --suffix "_quotesum" \
    --type open_ended \
    --behaviors "context-focus" \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=quotesum.json" \
    --override_vector_path "1COMPLETED_RUNS/39_llama-3.1-8b_no-options_max_tokens/normalized_vectors/context-focus/vec_layer_${NUM}_Meta-Llama-3.1-8B-Instruct.pt"
done