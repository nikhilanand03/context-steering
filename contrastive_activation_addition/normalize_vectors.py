from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import torch as t
import os
import argparse

def normalize_vectors(model_size: str, is_base: bool, n_layers: int, use_latest: bool=False, use_mistral:bool=False, suffix:str=""):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors"+suffix)
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        non_existent_behaviors = []
        for behavior in ALL_BEHAVIORS:
            try:
                vec_path = get_vector_path(behavior, layer, get_model_path(model_size, is_base, use_latest, use_mistral), suffix=suffix)
                vec = t.load(vec_path)
                norm = vec.norm().item()
                vecs[behavior] = vec
                norms[behavior] = norm
                new_path = vec_path.replace("vectors", "normalized_vectors")
                new_paths[behavior] = new_path
            except:
                non_existent_behaviors.append(behavior)
                pass

        existent_behaviors = [item for item in ALL_BEHAVIORS if item not in non_existent_behaviors]

        print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in existent_behaviors:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in existent_behaviors:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_latest", action="store_true", default=False)
    parser.add_argument("--use_mistral", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--nlayers", type=int, default=32)

    args = parser.parse_args()

    # normalize_vectors("7b", True, 32, args.use_latest, args.use_mistral) # 7b-base (if it exists)
    normalize_vectors("7b", False, args.nlayers, args.use_latest, args.use_mistral, args.suffix) # 7b-chat model
    # normalize_vectors("13b", False, 36, args.use_latest, args.use_mistral) # 13b-chat (if it exists)
    