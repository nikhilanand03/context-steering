###### LLAMA-70B ##########################
## (Set MISTRAL_LIKE_MODEL in helpers.py to Llama70B)

python baseline_methods.py --use_mistral --suffix "_llama70b_baseline_nq" --override_dataset_path "test_dataset_open_ended_nq.json"
python baseline_methods.py --use_mistral --suffix "_llama70b_baseline_popqa" --override_dataset_path "test_dataset_open_ended_popqa.json"
python baseline_methods.py --use_mistral --suffix "_llama70b_baseline_triviaqa" --override_dataset_path "test_dataset_open_ended_triviaqa.json"

#############################################