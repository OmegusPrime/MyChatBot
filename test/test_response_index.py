import csv
import tempfile
import unittest
from pathlib import Path

from response_index import ResponseRetriever, build_response_index, normalize_prompt


class TestResponseIndex(unittest.TestCase):
    def test_exact_and_high_confidence_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            dataset.mkdir()
            with (dataset / "train.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dialog", "act", "emotion"])
                writer.writeheader()
                writer.writerow({
                    "dialog": repr([
                        "How do I reset my forgotten password?",
                        "Use the password reset link on the sign-in page.",
                    ]),
                    "act": "[]",
                    "emotion": "[]",
                })
                writer.writerow({
                    "dialog": repr(["Where is my order?", "Check the tracking page for its latest location."]),
                    "act": "[]",
                    "emotion": "[]",
                })

            index_path = root / "response_pairs.db"
            stats = build_response_index(dataset, index_path)
            self.assertEqual(stats["unique_pairs"], 2)
            retriever = ResponseRetriever(index_path)
            try:
                exact = retriever.retrieve("Where is my order?")
                self.assertIsNotNone(exact)
                self.assertIn("tracking", exact[0])
                fuzzy = retriever.retrieve("reset forgotten password please")
                self.assertIsNotNone(fuzzy)
                self.assertIn("password reset link", fuzzy[0])
                self.assertIsNone(retriever.retrieve("quantum zebra"))
            finally:
                retriever.close()

    def test_normalization(self):
        self.assertEqual(normalize_prompt("  Hello, WORLD!  "), "hello world")


if __name__ == "__main__":
    unittest.main()
