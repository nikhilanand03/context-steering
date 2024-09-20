## This script generates hallucination vectors using llama-3.1 and then uses these on the context-focus 
## open-ended dataset to observe the effect of reducing hallucinations for context-focus.

python generate_vectors.py --layers $(seq 0 31) --save_activations --use_latest --behaviors "hallucination"

python normalize_vectors.py --use_latest

# python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "hallucination"

# python analyze_vectors.py --skip_base --use_latest --behaviors "hallucination"

python prompting_with_steering.py --layers $(seq 0 31) --use_latest --multipliers -1 0 1 --type ab --behaviors "hallucination"

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "hallucination"

# WE SEE IN THE LAYER SWEEP THAT THE BEST LAYER IS THE 12TH LAYER (ESPECIALLY EFFECTIVE FOR M=-1, REDUCING HALLUCINATIONS)

python prompting_with_steering.py --layers 12 --use_latest --multipliers -2.0 -1.0 0 --type open_ended --behaviors "context-focus" --override_vector_behavior "hallucination"
# I want it to use the context-focus open-ended dataset and steer using the hallucination vector

python scoring.py --behaviors "context-focus"

python plot_results.py --layers 12 --multipliers -2.0 -1.0 0 --type open_ended  --behaviors "hallucination" --title "Layer 13 - Llama 2 7B Chat"
