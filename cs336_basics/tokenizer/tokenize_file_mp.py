from multiprocessing import Pool
from pathlib import Path
import pickle

import numpy as np
import regex as re
from tqdm import tqdm

from cs336_basics.tokenizer.tokenizer import Tokenizer

SPECIALS = ["<|endoftext|>"]
_SPLIT_RE = re.compile(
    "(" + "|".join(map(re.escape, sorted(SPECIALS, key=len, reverse=True))) + ")"
)

_tok: Tokenizer | None = None


def init_worker(vocab, merges, specials):
    global _tok
    _tok = Tokenizer(vocab, merges, specials)


def tokenize_chunk(doc: str) -> list[int]:
    assert _tok is not None
    return _tok.encode(doc)


def split_docs(text: str) -> list[str]:
    parts = _SPLIT_RE.split(text)
    docs = []
    for i in range(0, len(parts) - 1, 2):
        docs.append(parts[i] + parts[i + 1])
    if parts[-1]:
        docs.append(parts[-1])
    return docs


def tokenize(vocab, merges, file_path, output_path):
    text = Path(file_path).read_text()
    docs = split_docs(text)
    del text

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    with Pool(initializer=init_worker, initargs=(vocab, merges, SPECIALS)) as pool:
        for doc_ids in tqdm(pool.imap(tokenize_chunk, docs, chunksize=64), total=len(docs)):
            chunks.append(np.array(doc_ids, dtype=np.uint16))
    arr = np.concatenate(chunks)
    np.save(output_path, arr)


if __name__ == "__main__":
    with open("./outputs/bpe_owt/ints_to_tokens.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("./outputs/bpe_owt/merges.pkl", "rb") as f:
        merges = pickle.load(f)

    for split in ["valid", "train"]:
        tokenize(
            vocab=vocab,
            merges=merges,
            file_path=f"./data/owt_{split}.txt",
            output_path=f"./tokenized/owt_{split}.npy",
        )
