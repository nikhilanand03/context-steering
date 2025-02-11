##################################################
# run experiment with CAD decoding (with context) (Llama-8B)

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
 --input_file ../data/nq/nq_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'

##################################################