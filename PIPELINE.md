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
Make sure you have trained vocab and merges in the [outputs](outputs) directory. If not, train it via the
[train_bpe.py](cs336_basics/tokenizer/train_bpe.py).

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
