#!/bin/bash
eval “$(conda shell.bash hook)” 
conda create -y -n sllama_conda python=3.11 && conda activate /opt/conda/envs/sllama_conda && conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia &
# Navigate to the directory containing the project
cd ./contrastive_activation_addition  && pip install -r requirements_new.txt
pip install --upgrade langchain langchain-core requests charset-normalizer &
wait
echo “Setup completed!”