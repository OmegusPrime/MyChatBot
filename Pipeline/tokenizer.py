import collections
import json
from ingest import ingest
from re import compile, UNICODE
class Tokenizer:
    PRE_TOK = compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        UNICODE
    )
    def __init__(self):
        self.merges: dict[tuple, str] = {}
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
    @staticmethod
    def pre_tokenize(text: str):
        tokens = []
        for match in Tokenizer.PRE_TOK.finditer(text):
            word = match.group()
            chars = list(('Ġ' + word.lstrip(' ')) if word.startswith(' ') else word)
            if chars:
                tokens.append(chars)
        return tokens
    @staticmethod
    def get_pairs(word):
        return [(word[i],word[i+1]) for i in range(len(word)-1)]
    @staticmethod
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
            new_corpus.append(new_word)
        return new_corpus
    def train(self, texts, vocab_size: int = 10000, min_freq: int = 1024):
        corpus = []
        for text in texts:
            corpus.extend(Tokenizer.pre_tokenize(text))
        base_vocab: set[str] = set()
        for word in corpus:
            base_vocab.update(word)
        self.vocab = {ch: i for i, ch in enumerate(sorted(base_vocab))}
        num_merges = vocab_size - len(self.vocab)
        for step in range(num_merges):
            counts = Tokenizer.count_pairs(corpus)
            if not counts:
                break
            best_pair, best_freq = counts.most_common(1)[0]
            if best_freq < min_freq:
                break
            new_tokens = best_pair[0]+best_pair[1]
            self.merges[best_pair] = new_tokens
            self.vocab[new_tokens] = len(self.vocab)
            corpus = self.merge(corpus, best_pair)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    def encode_word(self, chars):
        word = list(chars)
        merge_keys = list(self.merges.keys())
        while len(word) > 1:
            pairs = Tokenizer.get_pairs(word)
            best = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: merge_keys.index(p),
                default=None
            )
            if best is None:
                break
            word = Tokenizer.merge([word], best)[
                0]
        return [self.vocab.get(tok, self.vocab.get('<unk>', 0)) for tok in word]
    def encode(self, texts):
        ids = []
        for word_chars in self.pre_tokenize(texts):
            ids.append(self.encode_word(word_chars))
        return ids
    def decode(self, ids):
        tokens = [self.inv_vocab.get(i, '') for i in ids]
        text = ''.join(tokens)
        return text.replace('Ġ', ' ').strip()
    def save(self, path):
        data = {
            'vocab': self.vocab,
            'merges': [list(k) + [v] for k, v in self.merges.items()]
        }
        with open(path, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent = 4)
    def load(self, path):
        with open(path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = {(row[0], row[1]): row[2] for row in data['merges']}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}


    @staticmethod  # BUG 10: missing @staticmethod — `self` was the first arg but it's not an instance method
    def train_from_ingest(data_dir, vocab_size: int = 10000, save_path: str = "bpe.json"):
        ingestor = ingest(data_dir)
        texts = [doc['text'] for doc in ingestor.ingest() if doc['text'].strip()]
        tokenizer = Tokenizer()
        tokenizer.train(texts, vocab_size=vocab_size)
        tokenizer.save(save_path)
        return tokenizer
