# Contextual Contrastive Activation Addition

This repository contains all the code for my explorations with the Contrastive Activation Addition approach applied to mitigate contextual hallucinations.

## Setup

When you start a new instance, this script must be run.
You can exclude Azure and Huggingface and do that manually.

```
git clone https://github.com/nikhilanand03/context-steering.git
huggingface-cli login
export AZURE_OPENAI_API_KEY=<KEY>; export AZURE_OPENAI_ENDPOINT=<ENDPT>

conda create --name sllama_conda
conda init
source ~/.bashrc
conda activate sllama_conda
conda install pip
conda install python=3.11
pip install -r requirements_new.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install --upgrade langchain langchain-core requests charset-normalizer
```

If there are issues with charset-normalizer:

```
pip uninstall charset-normalizer
pip install charset-normalizer==3.2.0
```

## Scripts

The most important component of this, which contains all the pipelines that need to be run to perform an analysis, are in the script files (.sh). Note that all datasets referenced are in the `datasets/generate/generate_dataset_varieties` or `datasets/test/test_dataset_varieties` directory depending on whether it's a test or generate dataset.

### Script_FULL_llama-3.1-8b.sh

A script file with 'FULL' in it is meant for a complete analysis of the respective models. Each of these files is split into 4 parts:

1. Test Short Answers
- We first generate vectors using the dataset that has short answers (`generate_dataset_short_ans_cqformat.json`). For example, a prompt would look like:

```
[INST]
Context: Chicago has 775 episodes.
Question: How many episodes does Chicago Season 7 have?
Options:
 (A) 775
 (B) 23
[/INST] (A)
```

- Next, we normalise the vectors obtained and plot the activations.
- Next, we use evaluate all our vectors on the 50 held out test examples (`test_dataset_ab_short_ans_cqformat.json`).
- We plot results and look at the plot to determine the best layer. We set it using `BESTLAYER=<layer>`.
- Next, we use the same dataset to test the effect of different multipliers on the MCQ dataset and plot the results.
- Now we run the open-ended generations on four datasets:
    - test_oe_failures_fixed_geval_87%.json (Failure cases for the open-ended generations)
    - test_oe_successes.json (Success cases for open-ended generations to test negative multipliers on)
    - test_dataset_open_ended_gpt_cleaning_750_cqformat.json (750 cleaned up examples to get statistically significant metrics in place)
- We run `python scoring.py --behaviors "context-focus"; python average_scores.py` to score all the open-ended generations found in the results folder.
- Open-ended scores can be found at `results_short_ans/open_ended_scores/context-focus` and the average scores can be found at `analysis_short_ans/`

2. Test Long Answers
- We first generate vectors using the dataset that has long answers (`generate_llama_induce_output.json`). For example, a prompt would look like:

```
[INST]
Context: Chicago has 775 episodes.
Question: How many episodes does Chicago Season 7 have?
Options:
 (A) Based on the context, Chicago has 775 episodes.
 (B) Notwithstanding the context, Chicago has 23 episodes.
[/INST] (A)
```

- Next, we normalise the vectors obtained and plot the activations.
- Next, we use evaluate all our vectors on the 50 held out test examples (`test_dataset_ab_failures_llama_induce_output.json`).
- We plot results and look at the plot to determine the best layer. We set it using `BESTLAYER=<layer>`.
- Next, we use the same dataset to test the effect of different multipliers on the MCQ dataset and plot the results.
- Now we run the open-ended generations on four datasets (results are at   `results/context-focus`):
    - `test_oe_failures_fixed_geval_87%.json` (Failure cases for the open-ended generations)
    - `test_oe_successes.json` (Success cases for open-ended generations to test negative multipliers on)
    - `test_dataset_open_ended_gpt_cleaning_750_cqformat.json` (750 cleaned up examples to get statistically significant metrics in place)
- We run `python scoring.py --behaviors "context-focus"; python average_scores.py` to score all the open-ended generations found in the results folder.
- Open-ended scores can be found at `results/open_ended_scores/context-focus` and the average scores can be found at `analysis/`

3. Test Baseline Methods
- If we run `python baseline_methods.py`, it will test the baseline methods using `test_oe_failures_fixed_geval_87%.json` dataset.
- Adding the `--long` attribute will ensure the baseline methods use the `test_dataset_open_ended_gpt_cleaning_750_cqformat.json` dataset.
- We don't add the `--use_mistral` attribute here.
- In the code file, you'll find these lines. You can change the constants to True if you want those versions to run and False if you don't.
    ```
    CAD = False
    NEG = False
    FNEG = False
    CAD_WO_BASELINE = True
    ```
- The meanings of each constant are listed below:
    - CAD: Context-Aware Decoding (alpha=0 and alpha=1)
    - CAD_WO_BASELINE: Context-Aware Decoding (alpha=1 only)
    - FNEG: Filtered negative decoding (alpha=6, thresh=0.01)
    - NEG: Negative decoding (alpha=1)
- The results are saved into `results/context-focus` and there is one file for each method.

4. Test Contrastive Steering Methods
- We apply a specific alpha and a threshold using the `--alpha` and `--threshold` attributes.
    - Alpha represents how the CAD parameter.
    - Threshold is the proportion of tokens that are filtered out.
- We run `python scoring.py --behaviors "context-focus"; python average_scores.py` to score all the open-ended generations found in the results folder.
- Note that if something is already scored it doesn't rescore it. If you want to rescore a particular results file, you delete the score file and rerun the above command.

5. Instruction Following Eval
- To perform an instruction following eval, we run the `prompting_with_steering` and `baseline_methods` python files, with the only difference that we are using the `--type if_eval` attribute.
- This ensures that the IF-Eval dataset is used to get results. 
- Once the result files are obtained for IFEval, these must be transferred to the `10instruction_following_eval/data` directory.
- We must then run `python json2jsonl.py input.json output.jsonl` to convert the output files to the jsonl format.
- Then, the full IFEval metric calculations can be done using `10instruction_following_eval/run.sh`.

6. Other Datasets
- This is to evaluate on the other datasets. It's quite straightforward how this works.

### Script_FULL_mistral-7b.sh

- We must extract all folders (analysis, results, activations, normalized_vectors, vectors, and _short_ans folders) to a different directory before running this script.
- Here, we are simply changing the model. So the main difference (compared to the previous script) is that in all the lines of code we have `--use_mistral` added as an attribute.
- The entire procedure is identical, except for the usage of a different model.
- Remember that `--use_mistral` isn't used only for mistral. It can be used for any Mistral-Like Model (meaning the model structure is similar in terms of the names of the components like model.model.layers, etc.)
- Before running this, we check `helpers.py` to ensure the constant `MISTRAL_LIKE_MODEL` is set to the respective model path we want. This will automatically change the model in all the program files if we use the `--use_mistral` parameter.
- Before running `scoring.py`, we must first copy the notebook `convert_750cq2scorable.ipynb` into the same directory as the results, and run the cells one by one. This will ensure convert the results to the scorable format, which includes the ground truth (`sub_answer`) as well.

### Script_FULL_gemma-2b.sh

- Here, the script is identical to the previous one. The only difference is the induced generations (used for the generate and test directories) are the datasets induced by the respective model.
- `MISTRAL_LIKE_MODEL` needs to be changed in `helpers.py` to Gemma-2-2b's model path, but that's about it.
- Other steps are run similar to mistral above.

### Want to try a new model?

- Use the same script as Mistral and Gemma. Change the `MISTRAL_LIKE_MODEL` parameter in `helpers.py` to a new model path of your choice.
- Now go to `baseline_methods.py`, `contrastive+steering.py`, `llama_wrapper.py`, and `utils/tokenize.py`, and wherever you see a condition set for one of the different models, you can add a new condition for your model. For example, in `tokenize.py` there are conditions for how each model is tokenized. In `llama_wrapper.py` there are conditions for the "instruction-end" string for each model.
- If you also want to use a different induce dataset (like a dataset that uses generations from the respective model as the memory option), then follow instructions in the next section.

## Datasets

- We have 3 folders here (generate, test and raw). 
- For the `generate/` directory, we can run `python descriptive_dataset.py` to generate the "induce" dataset.
    - Change the `model_name` variable to hold the model that we're interested in.
    - Add an additional condition to set the `END_TOK`, `ADD_FROM_POS`, `PAD_TOKEN_ID`, `pad_token` for the respective model.
- All the datasets are stored into the "varieties" folder and are aptly named.
