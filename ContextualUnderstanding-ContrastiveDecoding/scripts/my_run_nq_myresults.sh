##################################################
# run experiment with CAD decoding (with context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/nq/nq_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
#  --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --use_gold_ctx \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'

CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
 --input_file ../data/nq/nq_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
#  --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --use_gold_ctx \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'

# CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
#  --model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#  --input_file ../data/nq/nq_test.tsv \
#  --eval_method CAD \
#  --n_examples 5 \
# #  --ret_path ./data/retrieval/nq_contriever_results.jsonl \
#  --use_gold_ctx \
#  --bf16 \
#  --alpha 0.5 \
#  --alias 'nq-alpha-0.5'

##################################################
# run experiment with MICD decoding (with context)
CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/nq/nq_test.tsv \
 --eval_method CD \
 --n_examples 5 \
#  --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --use_gold_ctx \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
 --input_file ../data/nq/nq_test.tsv \
 --eval_method CD \
 --n_examples 5 \
#  --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --use_gold_ctx \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'

# CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
#  --model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#  --input_file ../data/nq/nq_test.tsv \
#  --eval_method CD \
#  --n_examples 5 \
# #  --ret_path ./data/retrieval/nq_contriever_results.jsonl \
#  --use_gold_ctx \
#  --bf16 \
#  --alpha 0.5 \
#  --alias 'nq-alpha-0.5'

##################################################