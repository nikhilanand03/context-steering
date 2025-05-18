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

# cd contrastive_activation_addition

# chmod +x script_nq_tqa_pqa_acrossmodels.sh
# ./script_nq_tqa_pqa_acrossmodels.sh

# tar -czvf all_results_llama70b_nq_popqa_tqa.tar.gz ./results_llama70b_NQ ./results_llama70b_PopQA ./results_llama70b_TriviaQA

###########################################

# # 9. Baseline methods on Llama-70B [NOT DONE!!]

# cd contrastive_activation_addition

# chmod +x script_baseline_methods_llama70b.sh
# ./script_baseline_methods_llama70b.sh

# tar -czvf all_results_cad_llama70b.tar.gz ./results_llama70b_baselines

###########################################

# 10. All datasets on Llama70B (PopQA, TriviaQA, NQ) with system prompt

# cd contrastive_activation_addition

# chmod +x script_sys_prompt_llama70b_all_datasets.sh
# ./script_sys_prompt_llama70b_all_datasets.sh

# tar -czvf all_results_sysprompt_llama70b_all_ds.tar.gz ./results_llama70b_NQ_sysprompt ./results_llama70b_PopQA_sysprompt ./results_llama70b_TriviaQA_sysprompt

###########################################

# 11. All datasets on Llama70B (PopQA, TriviaQA, NQ) with baseline method (CAD)

# cd contrastive_activation_addition

# chmod +x script_baseline_methods_llama70b_pqa_nq_tqa.sh
# ./script_baseline_methods_llama70b_pqa_nq_tqa.sh

# tar -czvf all_results_baselines_llama70b_all_ds.tar.gz ./results_llama70b_baseline_nq ./results_llama70b_baseline_popqa ./results_llama70b_baseline_triviaqa

###########################################

# 12. Run baseline methods on flan-T5

# cd ContextualUnderstanding-ContrastiveDecoding/scripts

# chmod +x my_run_nq_theirresults_flant5.sh
# ./my_run_nq_theirresults_flant5.sh

# tar -czvf results_baselines_flant5.tar.gz ../src/contrastive_decoding/results

###########################################

# 13. Run Reg-Cls on NQ/TriviaQA

# cd ContextualUnderstanding-ContrastiveDecoding/scripts

# bash my_run_tqa_nq_regcls.sh

# cd ..

# tar -czvf results_regcls_nq_tqa.tar.gz ./results_regcls_triviaQA ./results_regcls_NQ

###########################################

# 14. ConfiQA-QA Llama

# cd contrastive_activation_addition

# LAYER=12
# BESTMULT=3

# python prompting_with_steering.py \
#     --layers $LAYER \
#     --use_latest \
#     --multipliers $BESTMULT \
#     --type open_ended \
#     --confiqa \
#     --behaviors "context-focus" \
#     --suffix "_llama8b_confiqa_QA_1000" \
#     --sample 1000 \
#     --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-QA.json" \
#     --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

# tar -czvf results_confiqa_QA_llama8b.gz ./results_llama8b_confiqa_QA_1000

#########################################

# 15. ConfiQA-QA Mistral

cd contrastive_activation_addition

LAYER=13
BESTMULT=3

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers $BESTMULT \
    --type open_ended \
    --use_mistral \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_mistral7b_confiqa_QA_1000" \
    --sample 1000 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-QA.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_mistral-7b-0.3/normalized_vectors/context-focus/vec_layer_${LAYER}_Mistral-7B-Instruct-v0.3.pt"

tar -czvf results_confiqa_QA_llama8b.gz ./results_mistral7b_confiqa_QA_1000

#########################################