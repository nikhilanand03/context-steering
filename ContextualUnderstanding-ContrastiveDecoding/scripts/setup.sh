cd ..

conda env create -f environment.yml

cd lib

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .