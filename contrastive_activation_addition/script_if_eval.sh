# Assume normalised_vector contains the vectors we'd like to use for steering.
# Then we can directly run the below commands to generate steering outputs and contrastive steering outputs for the if_eval dataset

# Once you get the outputs, 
    # 1. Transfer them to 10instruction_following_eval, 
    # 2. Run the if_eval, 
    # 3. Manually make a table containing the scores of each method

# PARALLEL 1,2,3,4 ->
python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 --type if_eval --behaviors "context-focus"
python prompting_with_steering.py --layers 12 --use_latest --multipliers 2 --type if_eval --behaviors "context-focus"
python prompting_with_steering.py --layers 12 --use_latest --multipliers 1 --type if_eval --behaviors "context-focus"
python prompting_with_steering.py --layers 12 --use_latest --multipliers 3 --type if_eval --behaviors "context-focus"

# python contrastive+steering.py --layer 12 --multipliers 1 --type "if_eval"

python baseline_methods.py --type "if_eval"