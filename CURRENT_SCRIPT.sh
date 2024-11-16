# 1. HOTPOT OUTPUTS

# cd contrastive_activation_addition/datasets/raw/context-focus/new_datasets/generate_datasets

# chmod +x get_hotpot_outputs.sh
# ./get_hotpot_outputs.sh

############################################

# 2. MULTI_CONTEXT

cd contrastive_activation_addition

chmod +x script_FULL_multicontext_llama_8b.sh
./script_FULL_multicontext_llama_8b.sh

tar -czvf analysis.tar.gz ./analysis
tar -czvf results.tar.gz ./results

# OUTPUTS? DIR: contrastive_activation_addition; FOLDERS: analysis, normalized_vectors, results, vectors

############################################