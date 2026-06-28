import time
import pickle
import regex as re
from tokenizer import Tokenizer


def get_compression_ratio(tokenizer, text_path, special_tokens=None):
    with open(text_path, "r") as f:
        raw_text = f.read()

    pattern = re.compile("|".join(map(re.escape, special_tokens)))
    texts_iterator = re.splititer(pattern, raw_text)
    texts = [next(texts_iterator) for _ in range(100)]
    compression_ratios = []
    for text in texts:
        text_num_bytes = len(text.encode("utf-8"))
        encoded = tokenizer.encode(text)
        num_tokens = len(encoded)
        compression_ratios.append(text_num_bytes / num_tokens)
    return sum(compression_ratios) / len(compression_ratios)


def get_throughput(tokenizer, text_path, special_tokens=None):
    with open(text_path, "r") as f:
        raw_text = f.read()
    pattern = re.compile("|".join(map(re.escape, special_tokens)))
    texts_iterator = re.splititer(pattern, raw_text)
    sample = "".join([next(texts_iterator) for _ in range(100)])
    num_bytes = len(sample.encode("utf-8"))
    start = time.perf_counter()
    tokenizer.encode(sample)
    elapsed = time.perf_counter() - start

    throughput = num_bytes / elapsed

    return throughput


special_tokens = ["<|endoftext|>"]
with open("./outputs/bpe_owt/ints_to_tokens.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("./outputs/bpe_owt/merges.pkl", "rb") as f:
    merges = pickle.load(f)
owt_ratio = get_compression_ratio(
    tokenizer=Tokenizer(vocab, merges, special_tokens),
    text_path = "./data/owt_valid.txt",
    special_tokens = special_tokens,
)

with open("./outputs/bpe_tinystories/ints_to_tokens.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("./outputs/bpe_tinystories/merges.pkl", "rb") as f:
    merges = pickle.load(f)
tinystories_ratio = get_compression_ratio(
    tokenizer=Tokenizer(vocab, merges, special_tokens),
    text_path = "./data/TinyStoriesV2-GPT4-valid.txt",
    special_tokens = special_tokens,
)
owt_text_tinystories_tokenizer_ratio = get_compression_ratio(
    tokenizer=Tokenizer(vocab, merges, special_tokens),
    text_path = "./data/owt_valid.txt",
    special_tokens = special_tokens,
)

with open("./outputs/bpe_owt/ints_to_tokens.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("./outputs/bpe_owt/merges.pkl", "rb") as f:
    merges = pickle.load(f)
throughput = get_throughput(
    tokenizer = Tokenizer(vocab, merges, special_tokens),
    text_path = "data/owt_valid.txt",
    special_tokens = special_tokens,
)

print("Compression ratios:")
print(f"OWT: {owt_ratio}")
print(f"Tiny Stories: {tinystories_ratio}")
print(f"OWT text TinyStories tokenizer: {owt_text_tinystories_tokenizer_ratio}")

print(f"Throughput: {throughput} bytes/second")
