import json
import os
from statistics import mean
import argparse

def run_pipeline(suffix):
    results_dir = f"results{suffix}/open_ended_scores/context-focus"
    analysis_dir = f"analysis{suffix}"
    output_file = os.path.join(analysis_dir, "score_averages.txt")
    print(output_file)
    print(analysis_dir)
    print(results_dir)

    with open(output_file, 'w') as out_f:
        for filename in os.listdir(results_dir):
            print("File:",filename)
            if filename.endswith('.json'):
                file_path = os.path.join(results_dir, filename)
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                scores = [item['score'] for item in data]
                reading_ease = [item['reading_ease'] for item in data]
                num_words = [item['num_words'] for item in data]
                try:
                    perplexities = [item['perplexity'] for item in data]
                except:
                    pass
                repetition_info = [item['repetition_info'] for item in data]
                print(scores)
                print("hi")
                
                if scores:
                    avg_score = mean(scores)
                    avg_RE = mean(reading_ease)
                    avg_numwords = mean(num_words)
                    try:
                        avg_perplexity = mean(perplexities)
                    except:
                        pass

                    reps_per_item = []
                    for item in repetition_info:
                        num_reps = 0
                        for key in item:
                            num_reps+=item[key]
                        reps_per_item.append(num_reps)

                    avg_reps = mean(reps_per_item)
                    
                    out_f.write(f"{filename} Avg Score -> {avg_score:.2f}\n")
                    out_f.write(f"\t Avg Reading Ease -> {avg_RE:.2f}\n")
                    out_f.write(f"\t Avg #Words-> {avg_numwords:.2f}\n")
                    out_f.write(f"\t Repetition Info-> {repetition_info}\n")
                    out_f.write(f"\t Avg Repetitions-> {avg_reps:.2f}\n")
                    try:
                        out_f.write(f"\t Avg Perplexity-> {avg_perplexity:.2f}\n\n")
                    except:
                        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="")

    args = parser.parse_args()
    run_pipeline(args.suffix)