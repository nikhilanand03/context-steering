# 1. HOTPOT OUTPUTS

# cd contrastive_activation_addition/datasets/raw/context-focus/new_datasets/generate_datasets

# chmod +x get_hotpot_outputs.sh
# ./get_hotpot_outputs.sh

############################################

# 2. MULTI_CONTEXT

# cd contrastive_activation_addition

# chmod +x script_FULL_multicontext_llama_8b.sh
# ./script_FULL_multicontext_llama_8b.sh

# tar -czvf all_results.tar.gz ./normalized_vectors ./vectors ./analysis ./results

# OUTPUTS? DIR: contrastive_activation_addition; FOLDERS: analysis, normalized_vectors, results, vectors

############################################

# 3. NO OPTIONS

# cd contrastive_activation_addition

# chmod +x script_FULL_no_options.sh
# ./script_FULL_no_options.sh

# tar -czvf all_results.tar.gz ./normalized_vectors ./vectors ./analysis ./results

###########################################

# 4. TRANSFER NO OPTS TO NQ_SWAP

# cd contrastive_activation_addition

# chmod +x script_transfer_no_opts2nqswap.sh
# ./script_transfer_no_opts2nqswap.sh

# tar -czvf all_results.tar.gz ./analysis ./results

###########################################

# 5. SYSTEM PROMPT -> ALL_MODELS -> WITH OPTIONS

cd contrastive_activation_addition

chmod +x script_sys_prompt_nq_swap.sh
./script_sys_prompt_nq_swap.sh

tar -czvf all_results.tar.gz ./results_llama8b ./results_mistral7b ./results_llama70b

###########################################