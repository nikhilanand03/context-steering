# BASELINE METHODS ACROSS MODELS

###### LLAMA-3.1-8B ##########################

python baseline_methods.py --suffix "_llama8b_baseline_nq" --override_dataset_path "test_dataset_open_ended_nq.json"
python baseline_methods.py --suffix "_llama8b_baseline_popqa" --override_dataset_path "test_dataset_open_ended_popqa.json"
python baseline_methods.py --suffix "_llama8b_baseline_triviaqa" --override_dataset_path "test_dataset_open_ended_triviaqa.json"

#############################################

###### MISTRAL-7B ##########################
## (Set MISTRAL_LIKE_MODEL in helpers.py to Mistral7B)

# python baseline_methods.py --use_mistral --suffix "_mistral7b_baseline_nq" --override_dataset_path "test_dataset_open_ended_nq.json"
# python baseline_methods.py --use_mistral --suffix "_mistral7b_baseline_popqa" --override_dataset_path "test_dataset_open_ended_popqa.json"
# python baseline_methods.py --use_mistral --suffix "_mistral7b_baseline_triviaqa" --override_dataset_path "test_dataset_open_ended_triviaqa.json"

#############################################

###### LLAMA-70B ##########################
## (Set MISTRAL_LIKE_MODEL in helpers.py to Llama70B)

# python baseline_methods.py --use_mistral --suffix "_llama70b_baseline_nq" --override_dataset_path "test_dataset_open_ended_nq.json"
# python baseline_methods.py --use_mistral --suffix "_llama70b_baseline_popqa" --override_dataset_path "test_dataset_open_ended_popqa.json"
# python baseline_methods.py --use_mistral --suffix "_llama70b_baseline_triviaqa" --override_dataset_path "test_dataset_open_ended_triviaqa.json"

#############################################