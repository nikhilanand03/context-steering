# 1. Dynamic multiplier! Running this on the NQ-SWAP dataset for now.

# cd contrastive_activation_addition

# chmod +x script_dynamic_multiplier.sh
# ./script_dynamic_multiplier.sh

# tar -czvf all_results_dynmult_nqswap_llama8b.tar.gz ./results_dynamic_m

#############################################################

# 2. Run baseline methods on mistral7b (instruct) for triviaqa

# cd ContextualUnderstanding-ContrastiveDecoding/scripts

# bash my_run_nq_myresults_tqa.sh

# cd ..

# tar -czvf results_baselines_myinstructmodels_tqa.tar.gz ./results_cad_TQA ./results_CD_TQA

###########################################

# 3. ConfiQA-MC

cd contrastive_activation_addition

LAYER=12
BESTMULT=2

python prompting_with_steering.py \
    --layers $LAYER \
    --use_latest \
    --multipliers $BESTMULT \
    --type open_ended \
    --confiqa \
    --behaviors "context-focus" \
    --suffix "_llama8b_confiqa_MC_1000" \
    --sample 1000 \
    --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_oe_ConFiQA-MC.json" \
    --override_vector_path "1COMPLETED_RUNS/FULL_ANALYSIS_Llama-3.1-8b/normalized_vectors/context-focus/vec_layer_${LAYER}_Meta-Llama-3.1-8B-Instruct.pt"

tar -czvf results_confiqa_MC_llama8b.gz ./results_llama8b_confiqa_MC_1000

###########################################