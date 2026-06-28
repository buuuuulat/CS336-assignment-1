import pickle

with open("./outputs/bpe_owt/ints_to_tokens.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
