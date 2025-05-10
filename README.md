# ContextFocus: Balancing Context and Memory in Large Language Models

This is a project that uses various approaches to improve focus on the context in large language models. The folders are described below:

## Approaches
- `contrastive_activation_addition`: This is the main folder, containing our experiments using activation steering (Contrastive Activation Addition) to steer large language models towards the context. The README inside this folder explains how to run the scripts to recreate the results.
- `entropy_decoding`: This folder contains the entropy decoding approach and some of the results obtained with the memotrap dataset.
- `multitoken_entropy_decoding`: Here, the entropy decoding approach was applied on a few multitoken examples.
- `baseline_comparison`: All baseline methods (including CAD, Negative Decoding, Filtered Negative Decoding) were tried out on the same multitoken examples to compare the results.

## Preliminary Analyses
- `tutorials`: The initial explorations and the basics of using LLMs for decoding are documented here.
- `context_aware_decoding`: This folder contains the code for our context-aware decoding approach and experiments with the same.
- `my_refusal_code`: I attempted to recreate the paper "Refusal is mediated by a single direction" from scratch here. I also have some code that relies on dependencies imported from the official repository, to recreate the steering approach as they had done it. 

## Miscellaneous Folders
- `auto_evaluation`: This contains some program files useful for performing GPT evaluations using Langchain and OpenAI.
- `instruction_following_eval`: This contains the code for performing an IFEval on the outputs of any model or pipeline. Instructions on how to use it are enclosed in the README of the respective folder. (We use this in contrastive activation addition approach)
