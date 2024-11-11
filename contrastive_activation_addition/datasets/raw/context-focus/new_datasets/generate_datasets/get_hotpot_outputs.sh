# cd contrastive_activation_addition/datasets/raw/context-focus/new_datasets/making_generate_datasets
curl -o hotpot_training_data.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
python filter_hallucinations_optimised.py

# This should generate the outputs file and store it in the same folder
# Download this back and we can then perform analyses locally. Then we can generate the synthetic dataset using that.
# So we probably need max 5k points in the synthetic dataset 
# but Im running on all 900k because maybe out of those 900k Ill get 5k points of hallucinations.
# synthetic dataset can probably be stored in github and we can use it to generate the vector and do the full pipeline.
# we frame the dataset according to the behaviour we want to encourage and the behaviour we dont want to persist.

# can do this locally...
# python analyse_outputs.py # this will generate the mistakes output file
