# 1. Running llama-3.1-8b on the nq dataset now

# cd contrastive_activation_addition

# chmod +x script_nq_dataset.sh
# ./script_nq_dataset.sh

# tar -czvf all_results4.tar.gz ./results_llama8b_NQ

#############################################################

# 2. TriviaQA dataset!

# cd contrastive_activation_addition

# chmod +x script_trivia_qa.sh
# ./script_trivia_qa.sh

# tar -czvf all_results4.tar.gz ./results_llama8b_TriviaQA

#############################################################

# 3. Baseline methods on Mistral-7B (Uncomment Part 2 in that code)

# cd contrastive_activation_addition

# chmod +x script_baseline_methods_all_models.sh
# ./script_baseline_methods_all_models.sh

# tar -czvf all_results_cad_mistral7b.tar.gz ./results_mistral7b_baselines

#############################################################

# 3. Baseline methods on Llama-8B (Uncomment Part 1 in that code)

# cd contrastive_activation_addition

# chmod +x script_baseline_methods_all_models.sh
# ./script_baseline_methods_all_models.sh

# tar -czvf all_results_cad_llama8b.tar.gz ./results_llama8b_baselines

#############################################################