# 1. System prompt methods on Llama-8B (Uncomment Part 1 in that code)

cd contrastive_activation_addition

chmod +x script_sys_prompt_all_datasets.sh
./script_sys_prompt_all_datasets.sh

tar -czvf all_results_sysprompt_llama8b_all_datasets.tar.gz ./results_llama8b_NQ_sysprompt ./results_llama8b_PopQA_sysprompt ./results_llama8b_TriviaQA_sysprompt

#############################################################

# 2. System prompt methods on Mistral-7B (Uncomment Part 2 in that code)

# cd contrastive_activation_addition

# chmod +x script_sys_prompt_all_datasets.sh
# ./script_sys_prompt_all_datasets.sh

# tar -czvf all_results_sysprompt_mistral7b_all_datasets.tar.gz ./results_mistral7b_NQ_sysprompt ./results_mistral7b_PopQA_sysprompt ./results_mistral7b_TriviaQA_sysprompt

#############################################################
