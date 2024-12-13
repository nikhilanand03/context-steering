# 1. QuoteSum dataset! Using no-options vector to generate.

cd contrastive_activation_addition

chmod +x script_llama318b_quotesum_from_nqswap.sh
./script_llama318b_quotesum_from_nqswap.sh

tar -czvf all_results2.tar.gz ./results_quotesum

#############################################################