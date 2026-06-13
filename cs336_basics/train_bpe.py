import regex as re
from collections import Counter
from multiprocessing import Pool

from cs336_basics.pretokenization_example import find_chunk_boundaries

pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def count_pretokens_in_chunk(path: str, start: int, end: int, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    pretokens_counter = Counter()
    with open(path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        special_tokens = sorted(special_tokens, key=len,
                                reverse=True)  # Remove intersection possibility between special tokens
        split_pattern = "|".join(map(re.escape, special_tokens))
        segments = re.split(split_pattern, chunk)
    else:
        segments = [chunk]

    for segment in segments:
        matches = pattern.finditer(segment)
        pretokens_counter.update([tuple(bytes([b]) for b in match.group().encode("utf-8")) for match in matches])

    return pretokens_counter  # dict[tuple[bytes, ...], int]


def init_vocab(special_tokens: list[str], vocab_size: int) -> dict[int, bytes]:
    assert len(special_tokens) + 256 <= vocab_size

    vocab = {i: bytes([i]) for i in range(256)}
    next_i = 256
    for special_token in special_tokens:
        vocab[next_i] = special_token.encode("utf-8")
        next_i += 1

    return vocab  # First 256 bytes + special tokens


def count_words(
        input_path: str,
        special_tokens: list[str],
        chunking_num_processes: int,
        split_special_token: bytes
) -> Counter[tuple[bytes, ...]]:
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, chunking_num_processes, split_special_token)

    # Parallelize
    params = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=chunking_num_processes) as pool:
        partial_counters = pool.starmap(count_pretokens_in_chunk, params, chunksize=1)
    pretokens_counter = sum(partial_counters, start=Counter())
    return pretokens_counter


def train_byte_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        chunking_num_processes: int,
        split_special_token=b"<|endoftext|>"
):
    vocab = init_vocab(special_tokens, vocab_size)  # Initialize vocab

    pretokens_counter = count_words(input_path, special_tokens, chunking_num_processes,
                                    split_special_token)  # Count all words

    # TODO: step 2 - merges

    # return vocab, merges


if __name__ == '__main__':
    pass
