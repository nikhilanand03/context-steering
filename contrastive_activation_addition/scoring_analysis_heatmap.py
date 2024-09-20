import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def plot_steering():
    files = os.listdir("results/open_ended_scores/context-focus")
    
    # print(files)
    score_lists = {}
    for file_name in files:
        if file_name.startswith("contrastive+"):
            continue
        mult = file_name[file_name.index("multiplier=")+11:file_name.index("behavior=context-focus")-1]
        file_path = os.path.join("results/open_ended_scores/context-focus", file_name)
        
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                li = []
                for item in data:
                    li.append(item['score'])
                score_lists[mult] = li
        
    if score_lists=={}:
        return

    data_list = [score_lists[key] for key in sorted(score_lists.keys(), key=float)]
    # print(data_list)

    plt.figure(figsize=(15, 8))
    sns.heatmap(data_list, cmap='YlOrRd', cbar_kws={'label': 'Faithfulness'})

    plt.yticks(range(len(score_lists)), sorted(score_lists.keys(), key=float), rotation=0)
    plt.xlabel('Test Example #',size=25)
    plt.ylabel('Multiplier',size=25)
    plt.title('Heatmap of Values',size=25)

    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig('analysis/steering_scoring_heatmap.png', dpi=300, bbox_inches='tight')

    plt.show()

def plot_contrastive_steering():
    files = os.listdir("results/open_ended_scores/context-focus")
    score_lists = {}
    for file_name in files:
        if not file_name.startswith("contrastive+"):
            continue
        mult = file_name[file_name.index("_mult=")+6:file_name.index(".json")]
        file_path = os.path.join("results/open_ended_scores/context-focus", file_name)
        
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                li = []
                for item in data:
                    li.append(item['score'])
                score_lists[mult] = li
            
    if score_lists=={}:
        return
            
    data_list = [score_lists[key] for key in sorted(score_lists.keys(), key=float)]

    plt.figure(figsize=(15, 8))
    sns.heatmap(data_list, cmap='YlOrRd', cbar_kws={'label': 'Faithfulness'})

    plt.yticks(range(len(score_lists)), sorted(score_lists.keys(), key=float), rotation=0)
    plt.xlabel('Test Example #',size=25)
    plt.ylabel('Multiplier',size=25)
    plt.title('Heatmap of Values',size=25)

    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(f'analysis/contrastive_steering_scoring_heatmap.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__=="__main__":
    plot_steering()
    plot_contrastive_steering()