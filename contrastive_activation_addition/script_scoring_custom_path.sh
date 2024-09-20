# With this script command, we can score a totally custom path (directory) in which several results are contained.
# They are scored in the same directory adding "/open_ended_scores_path" to the path.

# python scoring.py --behaviors "context-focus" --custom_path_dir "1COMPLETED_RUNS/10llama-3.1-8b-manually-scored/results/context-focus"

# Scoring with different num_times values; to check for the best multiplier.
python scoring.py --num_times 3 --behaviors "context-focus" --custom_path_dir "1COMPLETED_RUNS/20llama-3.1-8b-evaluation-correlations/results/context-focus"
python scoring.py --num_times 1 --behaviors "context-focus" --custom_path_dir "1COMPLETED_RUNS/20llama-3.1-8b-evaluation-correlations/results/context-focus"
python scoring.py --num_times 5 --behaviors "context-focus" --custom_path_dir "1COMPLETED_RUNS/20llama-3.1-8b-evaluation-correlations/results/context-focus"