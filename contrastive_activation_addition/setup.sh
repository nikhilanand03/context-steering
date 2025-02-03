# run first two lines, then comment them out and can run the script
# git clone https://github.com/nikhilanand03/context-steering.git
# cd context-steering/contrastive_activation_addition

huggingface-cli login
conda create --name sllama_conda; conda init; source ~/.bashrc; conda activate sllama_conda
conda install pip; conda install python=3.11
pip install -r requirements_new.txt

# Both below commands can be run in parallel
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install --upgrade langchain langchain-core requests charset-normalizer

# don't forget to activate sllama_conda once it's done.

cd contrastive_activation_addition/datasets/test/context-focus/test_dataset_varieties
FILENAME="test_dataset_open_ended_new_triviaqa.json"
wget --no-check-certificate -O "$FILENAME" "https://files.slack.com/files-pri/T23RE8G4F-F08C5FF7N64/download/test_dataset_open_ended_new_triviaqa.json?origin_team=E23RE8G4F"