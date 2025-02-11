import shutil  
import os  

from pathlib import Path  
  
# Get the absolute path to the script file  
script_path = Path(__file__).resolve()  
print(f"The absolute path of the script is: {script_path}")  
  
# If you want just the directory containing the script  
script_dir = script_path.parent  
print(f"The script is located in: {script_dir}")  

# Define the paths  
local_utils_path = "src/contrastive_decoding/lib/transformers/utils.py"  
library_utils_path = "transformers-4.34.0/src/transformers/generation/utils.py"  
  
local_mistral_path = "src/contrastive_decoding/lib/transformers/modeling_mistral.py"  
library_mistral_path = "transformers-4.34.0/src/transformers/models/mistral/modeling_mistral.py"  
  
# Function to replace files  
def replace_file(source, destination):  
    try:  
        source=os.path.join(script_dir,source)
        destination=os.path.join(script_dir,destination)
        print(os.path.exists(destination))
        shutil.copyfile(source, destination)  
        print(f"Successfully replaced {os.path.basename(destination)}")  
    except Exception as e:  
        print(f"Failed to replace {os.path.basename(destination)}: {e}")  
  
# Replace the utils.py file  
replace_file(local_utils_path, library_utils_path)  
  
# Replace the modelling_mistral.py file  
replace_file(local_mistral_path, library_mistral_path)  