# Train and run pipeline

## Download and prepare dataset

### 1. Download and unzip

```shell
mkdir -p data
cd data

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 2. Tokenize it

Firstly, edit the [tokenize_file_mp.py](cs336_basics/tokenizer/tokenize_file_mp.py) file to use actual paths.

The trained vocab and merges are **not** shipped in this repo — the [outputs](outputs) directory is gitignored, so
you have to build them yourself with [train_bpe.py](cs336_basics/tokenizer/train_bpe.py):

```shell
uv run cs336_basics/tokenizer/train_bpe.py
```

Its `__main__` block is set up for OpenWebText (`vocab_size=32000`, reads `./data/owt_train.txt`, writes to
`./outputs/bpe_owt/`). For the TinyStories tokenizer that `experiments.py`, `testing.py` and
`generate_text.py --tokenizer ./outputs/bpe_tinystories` expect, adjust the input path, `vocab_size` and the two
save paths accordingly.

Then, run:

```shell
uv run cs336_basics/tokenizer/tokenize_file_mp.py
```

Now you have the [tokenized](tokenized) directory with files in .npy format.

If using Runpod, you may want to move it from the network to the local disk:

```shell
df -h /workspace  # check if workspace dir is a web storage
df -h /dev/shm
cp /workspace/tokenized/*.npy /dev/shm/
```

### 3. Train the model
Edit the [train_config.yaml](configs/train_config.yaml) if necessary and run:

```shell
uv run cs336_basics/train.py
```

### 4. Results
Results will be saved to the [runs](runs) directory
