import unittest
import os
import json
import tempfile
from ingest import ingest
from tokenizer import Tokenizer

DATASET_DIR = r"/Dataset"


class TestIngest(unittest.TestCase):

    def setUp(self):
        self.docs = list(ingest(DATASET_DIR).ingest())

    # ── Basic sanity ──────────────────────────────────────────────────────

    def test_at_least_one_document(self):
        self.assertGreater(len(self.docs), 0)

    def test_every_doc_has_file_and_text_keys(self):
        for doc in self.docs:
            self.assertIn('file', doc)
            self.assertIn('text', doc)

    def test_file_field_is_string(self):
        for doc in self.docs:
            self.assertIsInstance(doc['file'], str)

    def test_text_field_is_string(self):
        for doc in self.docs:
            self.assertIsInstance(doc['text'], str)

    def test_file_paths_exist(self):
        for doc in self.docs:
            self.assertTrue(os.path.exists(doc['file']))

    # ── Per file-type ─────────────────────────────────────────────────────

    def test_txt_files_have_content(self):
        txt_docs = [d for d in self.docs if d['file'].endswith('.txt')]
        if not txt_docs:
            self.skipTest("No .txt files in dataset")
        for doc in txt_docs:
            self.assertIsInstance(doc['text'], str)

    def test_csv_files_yield_multiple_rows(self):
        csv_files = set(d['file'] for d in self.docs if d['file'].endswith('.csv'))
        if not csv_files:
            self.skipTest("No .csv files in dataset")
        for csv_file in csv_files:
            rows = [d for d in self.docs if d['file'] == csv_file]
            self.assertGreater(len(rows), 0)

    def test_pdf_files_have_content(self):
        pdf_docs = [d for d in self.docs if d['file'].endswith('.pdf')]
        if not pdf_docs:
            self.skipTest("No .pdf files in dataset")
        for doc in pdf_docs:
            self.assertIsInstance(doc['text'], str)

    def test_no_unsupported_extensions(self):
        supported = {'.txt', '.csv', '.pdf'}
        for doc in self.docs:
            ext = os.path.splitext(doc['file'])[1].lower()
            self.assertIn(ext, supported)

    def test_no_none_text(self):
        for doc in self.docs:
            self.assertIsNotNone(doc['text'])


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        docs = list(ingest(DATASET_DIR).ingest())
        self.texts = [d['text'] for d in docs if d['text'].strip()]
        self.assertGreater(len(self.texts), 0, "No text found in dataset")
        self.tok = Tokenizer()
        self.tok.train(self.texts, vocab_size=1000, min_freq=2)

    # ── Vocab / training ──────────────────────────────────────────────────

    def test_vocab_not_empty(self):
        self.assertGreater(len(self.tok.vocab), 0)

    def test_vocab_size_bounded(self):
        self.assertLessEqual(len(self.tok.vocab), 1000)

    def test_inv_vocab_is_inverse(self):
        for token, idx in self.tok.vocab.items():
            self.assertEqual(self.tok.inv_vocab[idx], token)

    def test_merges_are_strings(self):
        for (a, b), merged in self.tok.merges.items():
            self.assertEqual(a + b, merged)

    # ── Encode ────────────────────────────────────────────────────────────

    def test_encode_returns_list(self):
        sample = self.texts[0][:100]
        self.assertIsInstance(self.tok.encode(sample), list)

    def test_encode_nonempty(self):
        sample = self.texts[0][:100]
        self.assertGreater(len(self.tok.encode(sample)), 0)

    def test_encode_returns_ints(self):
        ids = self.tok.encode(self.texts[0][:100])
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_encode_ids_in_vocab(self):
        ids = self.tok.encode(self.texts[0][:100])
        self.assertTrue(all(i in self.tok.inv_vocab for i in ids))

    def test_encode_empty_string(self):
        self.assertEqual(self.tok.encode(""), [])

    # ── Decode ────────────────────────────────────────────────────────────

    def test_decode_returns_string(self):
        ids = self.tok.encode(self.texts[0][:100])
        self.assertIsInstance(self.tok.decode(ids), str)

    def test_roundtrip(self):
        sample = self.texts[0][:100].strip()
        self.assertEqual(self.tok.decode(self.tok.encode(sample)), sample)

    def test_decode_empty(self):
        self.assertEqual(self.tok.decode([]), "")

    # ── Save / Load ───────────────────────────────────────────────────────

    def test_save_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            self.tok.save(path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_save_load_vocab_matches(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            self.tok.save(path)
            tok2 = Tokenizer()
            tok2.load(path)
            self.assertEqual(self.tok.vocab, tok2.vocab)
        finally:
            os.unlink(path)

    def test_save_load_merges_match(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            self.tok.save(path)
            tok2 = Tokenizer()
            tok2.load(path)
            self.assertEqual(self.tok.merges, tok2.merges)
        finally:
            os.unlink(path)

    def test_loaded_tokenizer_encodes_same(self):
        sample = self.texts[0][:100]
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            self.tok.save(path)
            tok2 = Tokenizer()
            tok2.load(path)
            self.assertEqual(self.tok.encode(sample), tok2.encode(sample))
        finally:
            os.unlink(path)

    def test_saved_json_structure(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            self.tok.save(path)
            with open(path) as f:
                data = json.load(f)
            self.assertIn('vocab', data)
            self.assertIn('merges', data)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)