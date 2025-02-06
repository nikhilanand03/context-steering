cd ..

pip install -r requirements.txt

wget -O transformers-4.34.0.tar.gz https://files.pythonhosted.org/packages/8d/c6/8bec511e1011dc8da93c1173af9ef988310a416657a320e9ffecf892dcc3/transformers-4.34.0.tar.gz
tar -xvzf transformers-4.34.0.tar.gz
cd transformers-4.34.0
pip install -e .

# NOW SWAP OUT utils.py from the local directory into the library directory
# Local: src/contrastive-decoding/lib/transformers/utils.py
# Library: transformers-4.34.0/src/transformers/generation/utils.py

# go to: transformers-4.34.0/src/transformers/models/mistral/modelling_mistral.py
# replace the file with: src/contrastive_decoding/lib/transformers/modelling_mistral.py
# (I added a condition at sliding_window)

cd .. # home directory of ContextualUnderstanding-ContrastiveDecoding

pip install sentencepiece
pip install jsonlines
pip install accelerate
pip install protobuf
pip install Levenshtein

### LLAMA

pip install --upgrade huggingface-hub