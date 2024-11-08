# run first two lines, then comment them out and can run the script
git clone https://github.com/nikhilanand03/context-steering.git
cd context-steering/contrastive_activation_addition

huggingface-cli login
conda create --name sllama_conda; conda init; source ~/.bashrc; conda activate sllama_conda
conda install pip; conda install python=3.11
pip install -r requirements_new.txt

# Both below commands can be run in parallel
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install --upgrade langchain langchain-core requests charset-normalizer