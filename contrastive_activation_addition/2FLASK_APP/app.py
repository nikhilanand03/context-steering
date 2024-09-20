import sys
import os
import socket
import matplotlib.pyplot as plt
import matplotlib.colors

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

ip_address = socket.gethostbyname(socket.gethostname())

from flask import Flask, render_template, request, url_for
from prompt_model import *

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Initialize the model with a default multiplier
steered_model = load_model(layer=12, multiplier=0.0)

print(f"Flask app running on: http://{ip_address}:6006")

def get_vector_path(layer):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    return os.path.join(parent_dir, "1COMPLETED_RUNS", "FULL_ANALYSIS_Llama-3.1-8b", 
        "normalized_vectors", "context-focus", f"vec_layer_{layer}_Meta-Llama-3.1-8B-Instruct.pt")

def get_vector(layer):
    return torch.load(get_vector_path(layer))
    
def value_to_color(value, cmap=plt.cm.RdBu, vmin=-25, vmax=25):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(value))
    return matplotlib.colors.to_hex(rgba)

def generate_colored_text(text, dot_products):
    colored_text = ""
    max_dist_from_zero = max([abs(x[1]) for x in dot_products])
    tokens = steered_model.tokenizer.encode(text)
    tokens = steered_model.tokenizer.batch_decode(torch.tensor(tokens).unsqueeze(-1))
    
    for idx, (_, value) in enumerate(dot_products):
        color = value_to_color(value, vmin=-1 * max_dist_from_zero, vmax=max_dist_from_zero)
        if len(tokens[idx].strip()) == 0:
            colored_text += " "
        else:
            colored_text += f'<span style="background-color: {color}; padding: 2px 5px; display: inline-block;">{tokens[idx].strip()}</span>'
    
    return colored_text

@app.route('/epic-intern-dev/nikhil-anand-3/tb', methods=['GET', 'POST'])
def index():
    global steered_model, current_multiplier
    colored_answer1 = ""
    colored_answer2 = ""
    colored_answer3 = ""
    context = ""
    question = ""
    multiplier1 = 2.0
    multiplier2 = 0.0
    multiplier3 = -2.0
    
    if request.method == 'POST':
        context = request.form['context']
        question = request.form['question']
        multiplier1 = float(request.form.get('multiplier1', 2.0))
        multiplier2 = float(request.form.get('multiplier2', 0.0))
        multiplier3 = float(request.form.get('multiplier3', -2.0))
        show_colors = 'show_colors' in request.form

        vec = get_vector(12).to(steered_model.device)
        
        # Generate answers using different multipliers
        # Generate answers using different multipliers
        change_multiplier(layer=12, multiplier=multiplier1, model=steered_model)
        steered_model.set_calc_dot_product_with_after_steer(12, vec)
        answer1 = prompt_with_steering(steered_model, context, question)
        dot_products1 = steered_model.get_dot_products_after_steer(12)
        colored_answer1 = generate_colored_text(answer1, dot_products1) if show_colors else answer1
        
        change_multiplier(layer=12, multiplier=multiplier2, model=steered_model)
        steered_model.set_calc_dot_product_with_after_steer(12, vec)
        answer2 = prompt_with_steering(steered_model, context, question)
        dot_products2 = steered_model.get_dot_products_after_steer(12)
        colored_answer2 = generate_colored_text(answer2, dot_products2) if show_colors else answer2
        
        # change_multiplier(layer=12, multiplier=multiplier3, model=steered_model)
        # steered_model.set_calc_dot_product_with_after_steer(12, vec)
        # answer3 = prompt_with_steering(steered_model, context, question)
        # dot_products3 = steered_model.get_dot_products_after_steer(12)
        # colored_answer3 = generate_colored_text(answer3, dot_products3) if show_colors else answer3
    
    return render_template('index.html', 
                           answer1=colored_answer1, answer2=colored_answer2,
                           current_multiplier1=multiplier1,
                           current_multiplier2=multiplier2,
                           context=context, question=question)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)
