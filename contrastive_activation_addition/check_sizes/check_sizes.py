import os
import json

def count_items_in_json(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        print(f"{filename}: {len(data)} items")
                    else:
                        print(f"{filename}: Not a list")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {filename}: {e}")

# Example usage
folder_path = "."  # Change this to the actual folder path
count_items_in_json(folder_path)