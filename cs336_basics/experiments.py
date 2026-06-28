import pickle
import regex as re
from tokenizer import Tokenizer

special_tokens = ["<|endoftext|>"]
pattern = re.compile("|".join(map(re.escape, special_tokens)))
with open("./outputs/bpe_tinystories/ints_to_tokens.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("./outputs/bpe_tinystories/merges.pkl", "rb") as f:
    merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges, special_tokens)

with open("./data/owt_valid.txt", "r") as f:
    raw_text = f.read()
text = next(re.splititer(pattern, raw_text))

print(len(tokenizer.encode(text)))
