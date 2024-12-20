################## PART 1 ##################

# python baseline_methods.py --suffix "_llama8b_baselines" --long
# Will update results folder with results from all baseline methods using our test dataset

# python scoring.py --behaviors "context-focus"; python average_scores.py

#############################################

################## PART 2 ##################
# Set the MISTRAL_LIKE_MODEL in helpers.py to Mistral

python baseline_methods.py --use_mistral --suffix "_mistral7b_baselines" --long
# python scoring.py --behaviors "context-focus"; python average_scores.py

#############################################

################## PART 3 ##################
# Set the MISTRAL_LIKE_MODEL in helpers.py to Llama70B

# python baseline_methods.py --use_mistral --suffix "_llama70b_baselines" --long
# python scoring.py --behaviors "context-focus"; python average_scores.py

#############################################