##################################################
# Run PopQA on CD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Llama-2-7b-chat-hf \
 --input_file ../data/1smaller_datasets/popqa/popqa_test.tsv \
 --ret_path ../data/1smaller_datasets/popqa/popqa_contriever_results.jsonl \
 --eval_method CD \
 --n_examples 5 \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --max_new_tokens 200 \
 --alias 'CD_PopQA_llama2'

##################################################

##################################################
# Run PopQA on CAD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Llama-2-7b-chat-hf \
 --input_file ../data/1smaller_datasets/popqa/popqa_test.tsv \
 --ret_path ../data/1smaller_datasets/popqa/popqa_contriever_results.jsonl \
 --eval_method CAD \
 --n_examples 5 \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --max_new_tokens 200 \
 --alias 'CAD_PopQA_llama2'

##################################################

##################################################
# Run PopQA on Reg-Cls

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Llama-2-7b-chat-hf \
 --input_file ../data/1smaller_datasets/popqa/popqa_test.tsv \
 --ret_path ../data/1smaller_datasets/popqa/popqa_contriever_results.jsonl \
 --eval_method vanilla \
 --n_examples 5 \
 --use_random_irr \
 --bf16 \
 --sample 1000 \
 --max_new_tokens 200 \
 --alias 'RegCls_PopQA_llama2'

##################################################