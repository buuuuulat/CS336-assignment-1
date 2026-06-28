import pickle
import numpy as np
from pathlib import Path

from tokenizer import Tokenizer

def tokenize(tokenizer, file_path, output_path, special_tokens=None):
    ids = []
    with open(file_path, "r") as f:
        for tid in tokenizer.encode_iterable(f):
            ids.append(tid)
    arr = np.array(ids, dtype=np.uint16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)


with open("./outputs/bpe_owt/ints_to_tokens.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("./outputs/bpe_owt/merges.pkl", "rb") as f:
    merges = pickle.load(f)
special_tokens = ["<|endoftext|>"]

tokenizer = Tokenizer(vocab, merges, special_tokens)
tokenize(
    tokenizer=tokenizer,
    file_path="./data/owt_train.txt",
    output_path="./tokenized/owt_train.npy"
)
