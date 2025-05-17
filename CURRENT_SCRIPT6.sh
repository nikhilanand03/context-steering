#############################################################

# 1. Steering on PopQA-Contriever (Mistral-7B)

# cd contrastive_activation_addition

# bash script_few_shot_pqa.sh

# tar -czvf all_results_few_shot_pqa.tar.gz ./results_mistral7b_PQA_Contriever_FS

#############################################################

# 2. Baselines (regcls,CAD,CD) on PopQA-Contriever (Mistral-7B)

# cd ContextualUnderstanding-ContrastiveDecoding/scripts

# bash my_run_popqa_cad_cd_regcls.sh

# cd ..

# tar -czvf results_baselines_popqa.gz ./results_regcls_popqa ./results_CD_popqa ./results_CAD_popqa

#############################################################

# 3. Running ConfiQA

cd contrastive_activation_addition

bash script_confiqa.sh

tar -czvf results_confiqa_llama8b.gz ./results_llama8b_confiqa_QA_1000 ./results_llama8b_confiqa_MR_1000 ./results_llama8b_confiqa_MC_1000

#############################################################