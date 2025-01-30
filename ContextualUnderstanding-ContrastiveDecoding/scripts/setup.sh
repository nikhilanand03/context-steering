cd ..

conda env create -f environment.yml

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

pip install sentencepiece