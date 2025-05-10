python scoring.py --behaviors "context-focus" --suffix "_llama70b_baseline_nq"
python average_scores.py --suffix "_llama70b_baseline_nq"

python scoring.py --behaviors "context-focus" --suffix "_llama70b_baseline_popqa"
python average_scores.py --suffix "_llama70b_baseline_popqa"

python scoring.py --behaviors "context-focus" --suffix "_llama70b_baseline_triviaqa"
python average_scores.py --suffix "_llama70b_baseline_triviaqa"

python scoring.py --behaviors "context-focus" --suffix "_llama70b_baselines"
python average_scores.py --suffix "_llama70b_baselines"