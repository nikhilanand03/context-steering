# For this, the normalised_vectors dir is a prereq. Also add results and analysis dirs in appropriate format.
# Then in contrastive+steering, set the appropriate desired value for alpha and thresh
# Then run the script and it should generate the required results and analysis plots

# TIP: - For each alpha, start with smallest threshold=0.0 and increase until outputs look reasonable. 
#      - Then run the full thing with that threshold.
#      - Go to the next alpha and repeat.

# RUN 1: alpha = 0.5, thresh = 0.0
# RUN 2: alpha = 4.0, thresh = 1e-3

python contrastive+steering.py --layer 12 --multipliers 1 2 3
python scoring.py --behaviors "context-focus"

python plot_results.py --layers 12 --multipliers 1 2 3 --type open_ended --method "contrastive-steer" --behaviors "context-focus" --title "Layer 12 - Llama 3.1 8B Instruct"

python scoring_analysis_heatmap.py