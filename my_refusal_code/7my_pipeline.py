import torch
import random
import json
import os
import argparse
import os

if os.getcwd() == "/home/user/nikhil/my_refusal_direction/pipeline":
    os.chdir("..")
print(os.getcwd())

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config

import transformers

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

print("Imported")

model_path = "qwen/qwen-1_8b-chat"
model_alias = os.path.basename(model_path)

cfg = Config(model_alias=model_alias, model_path=model_path)
model_base = construct_model_base(cfg.model_path)

print("Initialised model")

from pipeline.run_pipeline import *

import json
with open('pipeline/runs/qwen-1_8b-chat/direction_metadata.json', 'r') as file:
    data = json.load(file)

pos,layer = data['pos'],data['layer']
direction = torch.load("pipeline/runs/qwen-1_8b-chat/direction.pt")

print(pos,layer,direction.shape)

baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

print("Set hooks. Now evaluating completions into",cfg.artifact_path())

path_prefix = "pipeline/my_runs/qwen-1_8b-chat/completions/"
done_completions = os.path.isfile(path_prefix+"jailbreakbench_ablation_completions.json") and os.path.isfile(path_prefix+"jailbreakbench_actadd_completions.json") and os.path.isfile(path_prefix+ "jailbreakbench_baseline_completions.json")

if not done_completions:
    print("Making harmful completions.")
    for dataset_name in cfg.evaluation_datasets:
            generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
            generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
            generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)
else:
    print("Harmful already completed.")

# Note, we are act-adding with a negative 1 coefficient on the harmful dataset
# It answers harmful instructions in this case.

# HARMLESS: Here we generate and save completions for harmless dataset.
# We do this for baseline first, then with act_add using a coeff of +1.0 (induce refusal)

done_completions = os.path.isfile(path_prefix+"harmless_ablation_completions.json") and os.path.isfile(path_prefix+"harmless_actadd_completions.json") and os.path.isfile(path_prefix+ "harmless_baseline_completions.json")

if not done_completions:
    print("Making harmless completions.")
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
    
    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
    
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)

else:
    print("Already done with harmless completions.")