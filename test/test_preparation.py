import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main_and_eval import prepare_artifacts
from Pipeline.tokenizer import Tokenizer


class TestArtifactPreparation(unittest.TestCase):
    def test_tiny_dataset_builds_split_streams_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            artifacts = root / "artifacts"
            dataset.mkdir()
            long_turn = "This is a sufficiently long conversational sentence for token windows. " * 2
            for split in ("train", "validation", "test"):
                with (dataset / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["dialog", "act", "emotion"])
                    writer.writeheader()
                    writer.writerow({
                        "dialog": repr([long_turn, "A readable response should follow the prompt."]),
                        "act": "[]",
                        "emotion": "[]",
                    })

            paths = prepare_artifacts(dataset, artifacts, vocab_size=350)
            tokenizer = Tokenizer.from_pretrained(str(paths.tokenizer))
            train = np.load(paths.train_stream, allow_pickle=False)
            validation = np.load(paths.validation_stream, allow_pickle=False)
            test = np.load(paths.test_stream, allow_pickle=False)
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))

            self.assertGreater(len(train), 33)
            self.assertGreater(len(validation), 0)
            self.assertGreater(len(test), 0)
            self.assertTrue(paths.response_index.exists())
            self.assertLess(int(train.max()), tokenizer.vocab_size)
            self.assertEqual(metadata["record_counts"], {"train": 1, "validation": 1, "test": 1})

            original_fingerprint = metadata["dataset_fingerprint"]
            with (dataset / "train.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    repr(["A newly added prompt.", "A newly added response."]),
                    "[]",
                    "[]",
                ])
            prepare_artifacts(dataset, artifacts, vocab_size=350)
            refreshed = json.loads(paths.metadata.read_text(encoding="utf-8"))
            self.assertNotEqual(refreshed["dataset_fingerprint"], original_fingerprint)
            self.assertEqual(refreshed["record_counts"]["train"], 2)


if __name__ == "__main__":
    unittest.main()
