## Pipeline

The pipeline goes like this:

1. Run filter_hallucinations_optimised.py on the curl-imported hotpot_training dataset to extract 900k examples and to generate model_output for each example. (Output: question, answer, model_output)
2. Then run analyse_outputs.py to score each output based on the ground truth answer and question alone. This is used to filter out the times where the model makes a mistake wrt the ground truth.
3. Then run mistakes2generate_ds.py to convert the scores file to a new generate ds that simply contains the times where model makes a mistake, except here we also see the full dataset format including options, rag, etc.
4. Now we aren't done since our generate_ds isn't clean enough.
5. So we run create_supp_facts_dataset.py to create a new dataset with the supporting contexts alone (not all contexts) from the final generate dataset.
6. Now we run filter_generate_ds_further.py to use the supporting facts dataset to filter the dataset down. The items are basically scored and stored in scored_generate_dataset. The scores are either 0, 1 or 2 based on 3 categories. 2 means that the model stated "the context doesn't contain the answer". 1 means the model stated something contradicting the context. 0 means the model's output actually aligns with the context (there might've been a mistake, or it might be with context but against ground truth).
7. We run scored2shortlisted.py to generate the shortlisted dataset (filter out the 1s and 2s).

The shortlisted_generate_dataset.json is the one we copy into the generate directory and rename according to the typical nomenclature.