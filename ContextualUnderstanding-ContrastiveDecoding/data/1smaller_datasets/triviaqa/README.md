## Steps:

- First we convert into the convertible format (`convert2convertibleformat.py`). It saves contexts into a directory and creates the squad input format.
- Then we run `utils/convert2squad.py` (put the necessary parameters as well). The final dataset size should be 88k. All wiki contexts and search contexts in various pairs.
- `retain_one_ctx_w_ans.py`: Then post this we remove repeated questions, but keep the contexts where the answer is there. We take the first one where we find the answer. This could be a wiki context or a search context.
- `convert2FINAL.py` inputs the `deduplicated_kilt_triviaqa_subset_squad` and `converted_kilt_triviaqa_subset` json files, and we output the final dataset.