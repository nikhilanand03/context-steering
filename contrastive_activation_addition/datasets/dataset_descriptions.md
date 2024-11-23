## All Multicontext and No-Options Datasets

We really need to talk about all the datasets.

- `generate_dataset_multicontext_2500.json`: This was created by running the cleaning script whose pipeline is mentioned in `raw/new_datasets/generate_datasets/cleaning_pipeline.md`. Basically we did the pipeline on 2500x8 points, and there's a 10% success rate, leaving us with ~2000 points in total (250x8).
- `test_dataset_ab_multicontext_2500.json`: 100 points from the previous dataset were simply separated and kept.
- `generate_dataset_no_options.json`: The points were directly taken from `generate_dataset_multicontext_2500.json` except `multicontext2nooptions.py` was run on it to remove options and add system prompt in the "no-options" format.
- `test_dataset_oe_bestlayer_no_options.json`: This dataset is for getting the best layer in the "no-options" run. It is copied from `test_dataset_ab_multicontext_2500.json` directly except it's in open-ended format. We used the `remove_options` function in `common_funcs.py` to do this.
- `test_dataset_open_ended_version=hotpotqa_multicontext_failures_625.json`: This is the dataset where we ran the original cleaning pipeline (same as the first point) on another 625x8 examples and ended up with 625x8/10 = 500 egs due to the 10% success rate. This is used to do statistically significant failure testing on hotpotqa multicontext dataset.
- `test_dataset_open_ended_version=hotpotqa_multicontext.json`: This is the open-ended dataset obtained from contextbench. This isn't necessary natural failures and starts at 80% accuracy.