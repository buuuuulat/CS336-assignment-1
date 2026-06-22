import pickle

with open("./outputs/bpe_tinystories/merges.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
