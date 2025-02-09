##################################################
# Run NQ-SWAP on Reg-Cls (DEBUGGING)

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/nqswap/nq_swap_test.tsv \
 --eval_method vanilla \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alias 'nq-alpha-0.5' \
 --suffix '_regcls_nqswap_DEBUG' \
 --debug

##################################################

##################################################
# Run NQ-SWAP on Reg-Cls

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/nqswap/nq_swap_test.tsv \
 --eval_method vanilla \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alias 'nq-alpha-0.5' \
 --suffix '_regcls_nqswap'

##################################################

##################################################
# Run NQ-SWAP on CD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/nqswap/nq_swap_test.tsv \
 --eval_method CD \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5' \
 --suffix '_CD_nqswap'

##################################################

##################################################
# Run NQ-SWAP on CAD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/nqswap/nq_swap_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5' \
 --suffix '_CAD_nqswap'

##################################################