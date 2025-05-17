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

# cd ContextualUnderstanding-ContrastiveDecoding/scripts

# bash my_run_nqswap_cad_cd_regcls.sh

# cd ..

# tar -czvf results_nqswap_cad_cd_regcls.tar.gz ./results_regcls_nqswap ./results_CD_nqswap ./results_CAD_nqswap

###########################################

# 6. ConfiQA-MR

cd contrastive_activation_addition

LAYER=12
BESTMULT=2

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers $BESTMULT \
    --type open_ended \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_llama8b_confiqa_MR_1000" \
    --sample 1000 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MR.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

tar -czvf results_confiqa_MR_llama8b.gz ./results_llama8b_confiqa_MR_1000

##############################################