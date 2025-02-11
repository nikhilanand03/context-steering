#############################################################

# 1. Steering on PopQA-Contriever (Mistral-7B)

cd contrastive_activation_addition

bash script_few_shot_pqa.sh

tar -czvf all_results_few_shot_pqa.tar.gz ./results_mistral7b_PQA_Contriever_FS

#############################################################

# 2. Baselines (regcls,CAD,CD) on PopQA-Contriever (Mistral-7B)

# cd contrastive_activation_addition

# chmod +x script_few_shot_pqa.sh
# ./script_few_shot_pqa.sh

# tar -czvf all_results_few_shot_pqa.tar.gz ./results_mistral7b_PQA_Contriever_FS

#############################################################