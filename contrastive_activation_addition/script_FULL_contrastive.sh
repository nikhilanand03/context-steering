
python generate_vectors.py \
    --layers $(seq 10 14) \
    --save_activations \
    --use_latest \
    --behaviors "context-focus" \
    --override_dataset "generate_dataset_contrastive.json" \
    --contrastive

python normalize_vectors.py --use_latest

python plot_activations.py --layers $(seq 10 14) --use_latest --behaviors "context-focus" --no_options

######################

