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

cd contrastive_activation_addition

chmod +x script_FULL_no_options.sh
./script_FULL_no_options.sh

tar -czvf all_results.tar.gz ./normalized_vectors ./vectors ./analysis ./results

###########################################