# 1. QuoteSum dataset! Using no-options vector to generate.

# cd contrastive_activation_addition

# chmod +x script_llama318b_quotesum_from_nqswap.sh
# ./script_llama318b_quotesum_from_nqswap.sh

# tar -czvf all_results_quotesum_nqswapvector.tar.gz ./results_quotesum

#############################################################

# 2. System prompt methods on Mistral-7B (Uncomment Part 1 in that code)

# cd contrastive_activation_addition

# chmod +x script_sys_prompt_llama70b_mistral7b_all_datasets.sh
# ./script_sys_prompt_llama70b_mistral7b_all_datasets.sh

# tar -czvf all_results_sysprompt_mistral7b_all_datasets.tar.gz ./results_mistral7b_NQ_sysprompt ./results_mistral7b_PopQA_sysprompt ./results_mistral7b_TriviaQA_sysprompt

#############################################################

# 3. All datasets on Mistral-7B (PopQA, TriviaQA, NQ) with baseline method (CAD)

# cd contrastive_activation_addition

# chmod +x script_baseline_methods_mistral7b_pqa_nq_tqa.sh
# ./script_baseline_methods_mistral7b_pqa_nq_tqa.sh

# tar -czvf all_results_baselines_mistral7b_all_ds.tar.gz ./results_mistral7b_baseline_nq ./results_mistral7b_baseline_popqa ./results_mistral7b_baseline_triviaqa

#############################################################

# 4. Run baseline methods on mistral7b (instruct) for the NQ dataset

# cd ContextualUnderstanding-ContrastiveDecoding/scripts

# chmod +x my_run_nq_myresults_nq.sh
# ./my_run_nq_myresults_nq.sh

# cd ..

# tar -czvf results_baselines_myinstructmodels.tar.gz ./results_cad_NQ ./results_CD_NQ

###########################################

# 5. Run CD,CAD,Reg-Cls methods on mistral7b (instruct) for the NQ-SWAP dataset

cd ContextualUnderstanding-ContrastiveDecoding/scripts

chmod +x my_run_nqswap_cad_cd_regcls.sh
./my_run_nqswap_cad_cd_regcls.sh

cd ..

tar -czvf results_nqswap_cad_cd_regcls.tar.gz ./results_regcls_nqswap ./results_CD_nqswap ./results_CAD_nqswap

###########################################