import collections

from Pipeline import ingest
from re import compile, UNICODE
class Tokenizer:
    PRE_TOK = compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        UNICODE
    )
    def __init__(self):
        self.merges: dict[tuple, str] = {}
        self.vocab: dict[int, str] = {}
        self.inv_vocab: dict[str, int] = {}
    @staticmethod
    def pre_tokenize(text: str):
        tokens = []
        for match in Tokenizer.PRE_TOK.finditer(text):
            word = match.group()
            chars = list((word.lstrip(' ') if word.startswith(' ') else word))
            if chars:
                tokens.append(chars)
        return tokens
    @staticmethod
    def get_pairs(word):
        return [(word[i],word[i+1]) for i in range(len(word)-1)]
    def count_pairs(corpus):
        counts: collections.Counter = collections.Counter()
        for word in corpus:
            for pair in Tokenizer.get_pairs(word):
                counts[pair] += 1
        return counts
    @staticmethod
    def merge(corpus,pair):
        a, b = pair
        merged = a + b
        new_corpus = []
        for word in corpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(tuple(new_word))
        return new_corpus
