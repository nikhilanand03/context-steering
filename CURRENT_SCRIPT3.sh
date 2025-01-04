# 1. Dynamic multiplier! Running this on the NQ-SWAP dataset for now.

cd contrastive_activation_addition

chmod +x script_dynamic_multiplier.sh
./script_dynamic_multiplier.sh

tar -czvf all_results_dynmult_nqswap_llama8b.tar.gz ./results_dynamic_m

#############################################################