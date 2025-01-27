CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name google-t5/t5-11b \
 --input_file ../data/nq/nq_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
 --use_gold_ctx \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5' \
#  --ret_path ./data/retrieval/nq_contriever_results.jsonl \

CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name google-t5/t5-11b \
 --input_file ../data/nq/nq_test.tsv \
 --eval_method CD \
 --n_examples 5 \
 --use_gold_ctx \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'
#  --ret_path ./data/retrieval/nq_contriever_results.jsonl \