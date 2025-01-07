# 1. System prompt methods on Llama-8B

cd contrastive_activation_addition

chmod +x script_sys_prompt_all_datasets_llama8b.sh
./script_sys_prompt_all_datasets_llama8b.sh

tar -czvf all_results_sysprompt_llama8b_all_datasets.tar.gz ./results_llama8b_NQ_sysprompt ./results_llama8b_PopQA_sysprompt ./results_llama8b_TriviaQA_sysprompt

#############################################################