python generate_vectors.py --layers $(seq 0 31) --save_activations --use_latest --behaviors "context-focus"
# '--use_latest' changes it to Llama-3. Everything else in the steering code should still work as expected.

python normalize_vectors.py --use_latest

# <-- PARALLEL 1 --> #

python plot_activations.py --layers $(seq 0 31) --use_latest --behaviors "context-focus"
# Here, use_latest's only purpose is to find the respective filename (since model name is stored in the file)

python analyze_vectors.py --skip_base --use_latest --behaviors "context-focus"
# Again, use_latest is to find the correct vectors that we want to use and then analyze them

# </-- PARALLEL 1 --> #

# -- PARALLEL 2 -- #

python prompting_with_steering.py --layers $(seq 0 31) --use_latest --multipliers -1 0 1 --type ab --behaviors "context-focus"

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --behaviors "context-focus"

# </-- PARALLEL 2 --> #

# We don't need --use_latest here because the previous command saved results with default naming (7b,use_base=False)
# That means it uses defaults here too and plots according to those filenames

## HAVE TO REDO THIS SCRIPT. BUT FIRST MAKE SURE LLAMA-3.1-8B IS RETURNING (A) OR (B) FROM THE PROMPT (REFORMAT THE PROMPT TILL IT IS)
## SINCE IT'S CURRENTLY RETURNING C, THE PROMPT TEMPLATE NEEDS TO CHANGE. CAN ALSO CHANGE OPTIONS IN THE DATASET BEFORE THE NEXT RUN.
## ONCE YOU'RE SURE IT UNDERSTANDS THAT IT NEEDS TO ANSWER THE QUESTION, ONLY THEN THE METHOD WILL WORK.

# WE SEE IN THE LAYER SWEEP THAT THE BEST LAYER IS THE __TH LAYER ON AVERAGE (BETWEEN POS AND NEG)

# <-- NO NEED FOR PARALLELS --> #

python prompting_with_steering.py --layers 12 --use_latest --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus"

python plot_results.py --layers 12 --multipliers -3 -2 -1 0 1 2 3 --type ab --behaviors "context-focus"

# OPEN-ENDED

python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 --type open_ended --behaviors "context-focus"

python scoring.py --behaviors "context-focus"

python plot_results.py --layers 12 --multipliers -3 -2 -1 0 1 2 --type open_ended --method "steer" --behaviors "context-focus" --title "Layer 12 - Llama 3.1 8B Instruct"

python scoring_analysis_heatmap.py

## Note, if you want to try multiple values of alpha, change the value in the contrastive+steering file and run above lines multiple times.