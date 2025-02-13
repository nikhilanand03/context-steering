# This accounts for Reg-Cls, CAD and CD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Llama-2-7b-chat-hf \
 --input_file ../data/1smaller_datasets/triviaqa/triviaqa_test.tsv \
 --eval_method vanilla \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'RegCls_triviaQA_llama2'

##############################################################

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Llama-2-7b-chat-hf \
 --input_file ../data/1smaller_datasets/triviaqa/triviaqa_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'CAD_triviaQA_llama2'

##############################################################

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name meta-llama/Llama-2-7b-chat-hf \
 --input_file ../data/1smaller_datasets/triviaqa/triviaqa_test.tsv \
 --eval_method CD \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'CD_triviaQA_llama2'

##############################################################