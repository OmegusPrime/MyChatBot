import tempfile
import unittest
from pathlib import Path

from Pipeline.tokenizer import SPECIAL_TOKENS, Tokenizer


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = [
            "hello world",
            " leading space",
            "multiple   spaces",
            "punctuation: hello, world!",
            "newlines\nand\ttabs",
            "café",
            "नमस्ते",
            "emoji 🙂",
        ]
        cls.tokenizer = Tokenizer()
        cls.tokenizer.train(cls.samples, vocab_size=450, min_freq=1)

    def test_exact_roundtrips(self):
        for sample in self.samples:
            with self.subTest(sample=sample):
                encoded = self.tokenizer.encode(sample)
                self.assertEqual(self.tokenizer.decode(encoded, skip_special=False), sample)
                self.assertTrue(all(token_id in self.tokenizer.inv_vocab for token_id in encoded))

    def test_special_tokens_are_single_ids(self):
        text = "<user> hello<assistant> hi<eos>"
        encoded = self.tokenizer.encode(text)
        self.assertEqual(encoded[0], self.tokenizer.vocab["<user>"])
        self.assertIn(self.tokenizer.vocab["<assistant>"], encoded)
        self.assertEqual(encoded[-1], self.tokenizer.vocab["<eos>"])
        visible = self.tokenizer.decode(encoded, skip_special=True)
        self.assertFalse(any(token in visible for token in SPECIAL_TOKENS))

    def test_user_text_can_treat_special_markers_as_plain_text(self):
        encoded = self.tokenizer.encode("show <user> literally", allow_special=False)
        self.assertNotIn(self.tokenizer.vocab["<user>"], encoded)
        self.assertEqual(
            self.tokenizer.decode(encoded, skip_special=False),
            "show <user> literally",
        )

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_save_load_is_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(str(path))
            loaded = Tokenizer.from_pretrained(str(path))
            self.assertEqual(loaded.vocab, self.tokenizer.vocab)
            self.assertEqual(loaded.merges, self.tokenizer.merges)
            for sample in self.samples:
                self.assertEqual(loaded.encode(sample), self.tokenizer.encode(sample))


if __name__ == "__main__":
    unittest.main()
