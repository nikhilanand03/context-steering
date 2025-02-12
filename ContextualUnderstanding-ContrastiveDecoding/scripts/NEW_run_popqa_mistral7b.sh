##################################################
# Run PopQA on Reg-Cls

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/1smaller_datasets/popqa/popqa_test.tsv \
 --ret_path ./data/1smaller_datasets/popqa/popqa_contriever_results.jsonl \
 --eval_method vanilla \
 --n_examples 5 \
 --use_random_irr \
 --bf16 \
 --sample 1000 \
 --alias 'RegCls_PopQA_mistral'

##################################################

##################################################
# Run PopQA on CD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/1smaller_datasets/popqa/popqa_test.tsv \
 --ret_path ./data/1smaller_datasets/popqa/popqa_contriever_results.jsonl \
 --eval_method CD \
 --n_examples 5 \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'CD_PopQA_mistral'

##################################################

##################################################
# Run PopQA on CAD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/popqa/popqa_test.tsv \
 --ret_path ./data/popqa/popqa_contriever_results.jsonl \
 --eval_method CAD \
 --n_examples 5 \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'CAD_PopQA_mistral'

##################################################