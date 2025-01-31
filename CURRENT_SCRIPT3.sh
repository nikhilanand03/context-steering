# 1. Dynamic multiplier! Running this on the NQ-SWAP dataset for now.

# cd contrastive_activation_addition

# chmod +x script_dynamic_multiplier.sh
# ./script_dynamic_multiplier.sh

# tar -czvf all_results_dynmult_nqswap_llama8b.tar.gz ./results_dynamic_m

#############################################################

# 2. Run baseline methods on mistral7b (instruct) for triviaqa

cd ContextualUnderstanding-ContrastiveDecoding/scripts

chmod +x my_run_nq_myresults_tqa.sh
./my_run_nq_myresults_tqa.sh

tar -czvf results_baselines_myinstructmodels_tqa.tar.gz ../src/contrastive_decoding/results

###########################################