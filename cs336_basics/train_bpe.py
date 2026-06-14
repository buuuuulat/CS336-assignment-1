import regex as re
from collections import Counter, defaultdict
from multiprocessing import Pool

from cs336_basics.pretokenization_example import find_chunk_boundaries

pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
BYTES_TOKENS = [bytes([i]) for i in range(256)]


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
        pretokens_counter.update(tuple(BYTES_TOKENS[b] for b in match.group().encode("utf-8")) for match in matches)

    return pretokens_counter  # dict[tuple[bytes, ...], int]


def init_vocab(special_tokens: list[str], vocab_size: int) -> dict[int, bytes]:
    assert len(special_tokens) + 256 <= vocab_size
    vocab = {i: b for i, b in enumerate(BYTES_TOKENS)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab  # First 256 bytes + special tokens


def count_words(
        input_path: str,
        special_tokens: list[str],
        chunking_num_processes: int,
        num_chunks: int,
        split_special_token: bytes
) -> Counter[tuple[bytes, ...]]:
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)

    # Parallelize
    params = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=chunking_num_processes) as pool:
        partial_counters = pool.starmap(count_pretokens_in_chunk, params, chunksize=1)
    pretokens_counter = sum(partial_counters, start=Counter())
    return pretokens_counter


def count_pairs_and_index(
        pretokens_counter: Counter[tuple[bytes, ...]]
) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pairs_counter = Counter()
    pairs_to_pretokens = defaultdict(set)
    for pretoken, pretoken_freq in pretokens_counter.items():
        for left, right in zip(pretoken, pretoken[1:]):
            pair = (left, right)
            pairs_counter[pair] += pretoken_freq
            pairs_to_pretokens[pair].add(pretoken)
    return pairs_counter, pairs_to_pretokens


def single_merge(
        vocab: dict[int, bytes],
        pairs_counter: Counter[tuple[bytes, bytes]],
        pretokens_counter: Counter[tuple[bytes, ...]],
        pairs_to_pretokens: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
):
    max_pair = max(pairs_counter, key=lambda pair: (pairs_counter[pair], pair))  # Max by counts and lexicography
    vocab[len(vocab)] = b''.join(max_pair)

    for pretoken in list(pairs_to_pretokens[max_pair]):
        freq = pretokens_counter[pretoken]
        pairs = tuple(((a, b) for a, b in zip(pretoken, pretoken[1:])))
        for pair in pairs:
            pairs_counter[pair] -= freq
            if pairs_counter[pair] == 0:
                del pairs_counter[pair]
            pairs_to_pretokens[pair].discard(pretoken)

        i = 0
        new_pretoken = []
        while i < len(pretoken):
            if i + 1 < len(pretoken):
                pair = (pretoken[i], pretoken[i + 1])
                if pair == max_pair:
                    new_pretoken.append(b''.join(pair))
                    i += 2
                else:
                    new_pretoken.append(pair[0])
                    i += 1
            else:
                new_pretoken.append(pretoken[i])
                i += 1

        new_pretoken = tuple(new_pretoken)
        for pair in zip(new_pretoken, new_pretoken[1:]):
            pairs_counter[pair] += freq
            pairs_to_pretokens[pair].add(new_pretoken)

        del pretokens_counter[pretoken]
        pretokens_counter[new_pretoken] += freq

    del pairs_to_pretokens[max_pair]

    return max_pair


def train_byte_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        chunking_num_processes: int,
        num_chunks: int,
        split_special_token: bytes = b"<|endoftext|>"
):
    vocab = init_vocab(special_tokens, vocab_size)
    pretokens_counter = count_words(input_path, special_tokens, chunking_num_processes, num_chunks, split_special_token)
    pairs_counter, pairs_to_pretokens = count_pairs_and_index(pretokens_counter)
    merges = []

    c = 0
    while len(vocab) < vocab_size and pairs_counter:
        merge = single_merge(vocab, pairs_counter, pretokens_counter, pairs_to_pretokens)
        merges.append(merge)
        if c % 100 == 0:
            print(f"Step: {c} finished")
        c += 1

    return vocab, merges


if __name__ == '__main__':
    vocab, merges = train_byte_bpe(input_path="../data/owt_train.txt", vocab_size=10000,
                                   special_tokens=["<|endoftext|>"],
                                   chunking_num_processes=8, num_chunks=200, split_special_token=b"<|endoftext|>")
    print("VOCAB")
    for k, v in vocab.items():
        print(v)
