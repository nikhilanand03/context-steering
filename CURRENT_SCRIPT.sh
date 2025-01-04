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

# cd contrastive_activation_addition

# chmod +x script_sys_prompt_nq_swap.sh
# ./script_sys_prompt_nq_swap.sh

# tar -czvf all_results.tar.gz ./results_llama8b ./results_mistral7b ./results_llama70b

###########################################

# 6. POPQA -> LLAMA-8B

# cd contrastive_activation_addition

# chmod +x script_pop_qa.sh
# ./script_pop_qa.sh

# tar -czvf all_results.tar.gz ./results_llama8b_PopQA

###########################################

# 7. RUN ALL DATASETS (POPQA,NQ,TRIVIAQA) ON MISTRAL7B
# Uncomment mistral part of the script only

# cd contrastive_activation_addition

# chmod +x script_nq_tqa_pqa_acrossmodels.sh
# ./script_nq_tqa_pqa_acrossmodels.sh

# tar -czvf all_results.tar.gz ./results_mistral7b_NQ ./results_mistral7b_PopQA ./results_mistral7b_TriviaQA

###########################################

# 8. RUN ALL DATASETS (POPQA,NQ,TRIVIAQA) ON LLAMA70B
# Uncomment llama70B part of the script only

cd contrastive_activation_addition

chmod +x script_nq_tqa_pqa_acrossmodels.sh
./script_nq_tqa_pqa_acrossmodels.sh

tar -czvf all_results_llama70b_nq_popqa_tqa.tar.gz ./results_llama70b_NQ ./results_llama70b_PopQA ./results_llama70b_TriviaQA

###########################################