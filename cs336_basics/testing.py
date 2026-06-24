import pickle

from tokenizer import Tokenizer


with open("./outputs/bpe_tinystories/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("./outputs/bpe_tinystories/merges.pkl", "rb") as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
text = "Multilayer perceptron!"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Tokens:", [tokenizer.decode([t]) for t in encoded])
print("Decoded:", decoded)
