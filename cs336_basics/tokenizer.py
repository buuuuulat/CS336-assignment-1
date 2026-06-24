import math
import pickle
from typing import Iterable, Iterator

import regex as re


# noinspection PyAttributeOutsideInit
class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ) -> None:
        self.init_class(vocab, merges, special_tokens)

    def init_class(self, vocab, merges, special_tokens):
        self.ints_to_tokens = vocab
        self.tokens_to_ints = {v: k for k, v in self.ints_to_tokens.items()}
        self.merges_to_ids = {merges[i]: i for i in range(len(merges))}
        self.special_tokens = special_tokens
        if self.special_tokens:
            for st in self.special_tokens:
                st_b = st.encode("utf-8")
                if st_b not in self.tokens_to_ints:
                    self.tokens_to_ints[st_b] = len(self.tokens_to_ints)
                    self.ints_to_tokens[len(self.ints_to_tokens)] = st_b
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def from_files(
            self,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ) -> None:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        self.init_class(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        specials = set(self.special_tokens or [])
        for text in iterable:
            for segment in self._split_special(text):
                if segment in specials:
                    yield self.tokens_to_ints[segment.encode("utf-8")]
                else:
                    for pretoken in re.finditer(self.pattern, segment):
                        yield from self._encode_pretoken(pretoken.group())

    def decode(self, ids: list[int]) -> str:
        output = b''.join(self.ints_to_tokens[i] for i in ids)
        return output.decode("utf-8", errors="replace")

    def _split_special(self, text: str) -> Iterator[str]:
        if not self.special_tokens:
            yield text
            return
        specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(map(re.escape, specials)) + ")"
        yield from re.splititer(pattern, text)

    def _encode_pretoken(self, pretoken: str) -> Iterator[int]:
        pretoken_b = tuple(bytes([i]) for i in pretoken.encode("utf-8"))
        while len(pretoken_b) > 1:
            pairs = list(zip(pretoken_b, pretoken_b[1:]))
            best = min(pairs, key=lambda p: self.merges_to_ids.get(p, math.inf))
            if best not in self.merges_to_ids:
                break
            pretoken_b = self.merge(pretoken_b, best)
        for t in pretoken_b:
            yield self.tokens_to_ints[t]

    @staticmethod
    def merge(pretoken: tuple[bytes, ...], best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        output = []
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == best_pair:
                output.append(pretoken[i] + pretoken[i + 1])
                i += 2
            else:
                output.append(pretoken[i])
                i += 1
        return tuple(output)
