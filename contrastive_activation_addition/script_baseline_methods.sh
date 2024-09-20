python baseline_methods.py
# Will update results folder with results from all baseline methods using our test dataset

python scoring.py --behaviors "context-focus"
# Scores whatever it finds in the results folder with "*open_ended*" in the file name

python average_scores.py
# This gets avg score of each json file in a particular folder and stores to 