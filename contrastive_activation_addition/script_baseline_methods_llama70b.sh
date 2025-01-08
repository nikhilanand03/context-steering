################## PART 1 ##################
# Set the MISTRAL_LIKE_MODEL in helpers.py to Llama70B

python baseline_methods.py --use_mistral --suffix "_llama70b_baselines" --long
# python scoring.py --behaviors "context-focus"; python average_scores.py

#############################################