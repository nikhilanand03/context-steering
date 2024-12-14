# 1. Running llama-3.1-8b on the nq dataset now

# cd contrastive_activation_addition

# chmod +x script_nq_dataset.sh
# ./script_nq_dataset.sh

# tar -czvf all_results4.tar.gz ./results_llama8b_NQ

#############################################################

# 2. TriviaQA dataset!

cd contrastive_activation_addition

chmod +x script_trivia_qa.sh
./script_trivia_qa.sh

tar -czvf all_results4.tar.gz ./results_llama8b_TriviaQA

#############################################################