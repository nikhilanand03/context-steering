# Demo Website

## Setup

You need to run the following in the home directory (with the `requirements.txt` file) before running the app.

```
conda create --name sllama_conda
conda activate sllama_conda
conda install pip
conda install python=3.11
pip install -r requirements_new.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia (the new cuda that's compatible with our system)
pip install --upgrade langchain langchain-core requests charset-normalizer
```

## How to run?

```
python app.py
```

If you run into port availability issues:

```
lsof -i :6006
kill <PID>
```

Port 6006 is served here for the `nikhil-anand-3` job: `https://sensei-eks01.infra.adobesensei.io/epic-intern-dev/nikhil-anand-3/tb`
