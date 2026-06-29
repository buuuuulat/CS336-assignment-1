import pickle

from tokenizer import Tokenizer

with open("./outputs/bpe_owt/ints_to_tokens.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("./outputs/bpe_owt/merges.pkl", "rb") as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
text = "Hello my name is bob!"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Text:", text)
print("Encoded:", encoded)
print("Tokens:", [tokenizer.decode([t]) for t in encoded])
print("Decoded:", decoded)
