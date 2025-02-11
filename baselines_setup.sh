#!/bin/bash  
eval "$(conda shell.bash hook)" && conda create -y -n baselines_steering python=3.8 && conda activate /opt/conda/envs/baselines_steering
which python

echo "Sleeping for 5 seconds..."  
sleep 5
echo "Check output!"
git clone https://github.com/nikhilanand03/context-steering.git
cd context-steering/ContextualUnderstanding-ContrastiveDecoding
pip install -r devbox_req.txt
echo "Sleeping for 5 seconds..."  
sleep 5
echo "Check output!"


wget -O transformers-4.34.0.tar.gz https://files.pythonhosted.org/packages/8d/c6/8bec511e1011dc8da93c1173af9ef988310a416657a320e9ffecf892dcc3/transformers-4.34.0.tar.gz
tar -xzf transformers-4.34.0.tar.gz
rm transformers-4.34.0.tar.gz
cd transformers-4.34.0
pip install -e .
cd ..
echo "Sleeping for 5 seconds..."  
sleep 5
echo "Awake now!"
python transformers_mod.py
cd .. # home directory of ContextualUnderstanding-ContrastiveDecoding

pip install sentencepiece
pip install jsonlines
pip install accelerate
pip install protobuf
pip install Levenshtein

### LLAMA

pip install --upgrade huggingface-hub



