python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" --behaviors "hallucination"
# It generates vector for all 32 layers in 7b chat model, saving the activations of each layer, specifically for the hallucinations dataset.
# One activations file consists of the entire set of pos points, but for that specific layer. It's a tensor.
# One vec file contains the mean vector across the set of pos points for a given layer

python normalize_vectors.py
# This isn't actually necessary, since we only have one behavior

python plot_activations.py --layers $(seq 0 31) --model_size "7b" --behaviors "hallucination"
# The top-2 PCA plots for hallucination are plotted here. (analysis folder)

python analyze_vectors.py --skip_base --model_sizes "7b" --behaviors "hallucination"
# Usually plots similarity matrix of steering vectors across layers. 
# Also plots similarities of vectors between base and chat
# In this case, it'll plot whatever it 'can'. So just the similarity matrix for hallucination. (analysis folder)

python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --behaviors "hallucination"
# Earlier we used the MCQ (ab) dataset to generate vectors.
# Now, we use try steering using all 32 vectors and try to steer the model in 3 ways (-1,0,+1)
# We save the output probabilities of A and B in a new file in the results folder
# This is all done on the 50 held-out test egs.

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "hallucination"
# This should plot layer sweeps of hallucinations for the +1 and -1 multipliers (probabilities saved in results)
# This will give us the best layer as decided in the sweep

# WE SEE IN THE LAYER SWEEP THAT THE BEST LAYER IS THE 13TH LAYER ON AVERAGE (BETWEEN POS AND NEG)

python prompting_with_steering.py --layers 13 --multipliers -2 -1 -0.5 0 0.5 1 2 --type ab --behaviors "hallucination"
# Here, we only get results for steering layer 13, but we try to steer it by a larger amount (upto multiplier of 2)

python plot_results.py --layers 13 --multipliers -2 -1 -0.5 0 0.5 1 2 --type ab --behaviors "hallucination"
# Plots only for layer 13 but for all the multipliers for the MCQ dataset
# This will add a plot where the X-axis contains the multipliers and the y-axis shows the absolute accuracies
# Add the multiplier 0 to know the base accuracy

python prompting_with_steering.py --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 --type open_ended --behaviors "hallucination"
# Here, we use open-ended versions of the questions and the results are outputted.
# Note the test_dataset_open_ended.json in the datasets/test folder.
# The model uses the questions here (without options) as the input and returns the answer from the model
# These open-ended answers are thus stored in results.

python scoring.py --behaviors "hallucination"
# This does GPT-Eval for hallucination open-ended generations.

python plot_results.py --layers 13 --multipliers -2.0 -1.5 -1 0 1 --type open_ended  --behaviors "hallucination" --title "Layer 13 - Llama 2 7B Chat"
# This just plots the average scores obtained for each multiplier across the test prompts.
# Here, the plot is based on behavioral scores.
# When multiplier is positive, behavioural score is how frequently it hallucinates.
# When multiplier is negative, behavioural score is how frequently it answers truthfully.
# So we expect the graph to be bell shaped, increasing on either end of 0, if it is correct.