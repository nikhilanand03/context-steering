# 1. System prompt methods on Llama-8B

# cd contrastive_activation_addition

# chmod +x script_sys_prompt_all_datasets_llama8b.sh
# ./script_sys_prompt_all_datasets_llama8b.sh

# tar -czvf all_results_sysprompt_llama8b_all_datasets.tar.gz ./results_llama8b_NQ_sysprompt ./results_llama8b_PopQA_sysprompt ./results_llama8b_TriviaQA_sysprompt

#############################################################

# 2. All datasets on Llama-8B (PopQA, TriviaQA, NQ) with baseline method (CAD)

# cd contrastive_activation_addition

# chmod +x script_baseline_methods_all_models_pqa_nq_tqa.sh
# ./script_baseline_methods_all_models_pqa_nq_tqa.sh

# tar -czvf all_results_baselines_llama8b_all_ds.tar.gz ./results_llama8b_baseline_nq ./results_llama8b_baseline_popqa ./results_llama8b_baseline_triviaqa

#############################################################

# 3. Steering using the Few-shot approach (all datasets, Mistral model)

cd contrastive_activation_addition

bash script_few_shot_nq_tqa_pqa.sh

tar -czvf all_results_few_shot_nq_tqa_pqa.tar.gz ./results_mistral7b_NQ_FS ./results_mistral7b_TQA_FS ./results_mistral7b_PQA_BM25_FS

#############################################################