import torch

def get_cosine_sim(vec1_path, vec2_path):
    vec1 = torch.load(vec1_path)
    vec2 = torch.load(vec2_path)
    
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    dot_product = torch.dot(vec1, vec2)
    
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    
    cosine_sim = dot_product / (norm1 * norm2)
    
    return cosine_sim

if __name__=="__main__":
    path1 = "1COMPLETED_RUNS/14llama-3.1-8b-hallucination-vector-2-context-dataset/normalized_vectors/hallucination/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt"
    path2 = "1COMPLETED_RUNS/9llama-3.1-8b-test_cleaned_metric_fixed_scores_plotted/normalized_vectors/context-focus/vec_layer_12_Meta-Llama-3.1-8B-Instruct.pt"
    print(get_cosine_sim(path1,path2))