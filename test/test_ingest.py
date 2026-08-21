import csv
import tempfile
import unittest
from pathlib import Path

from Pipeline.ingest import Ingest, format_conversation, parse_serialized_turns


class TestConversationHelpers(unittest.TestCase):
    def test_serialized_adjacent_strings_become_turns(self):
        turns = parse_serialized_turns("['hello' 'there' \"general kenobi\"]")
        self.assertEqual(turns, ["hello", "there", "general kenobi"])

    def test_conversation_uses_canonical_roles_and_eos(self):
        self.assertEqual(
            format_conversation(["hello", "hi"]),
            "<user> hello\n<assistant> hi\n<eos>",
        )


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_csv(self, name, fieldnames, rows):
        with (self.root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_explicit_conversational_schemas(self):
        (self.root / "movie_lines.txt").write_text(
            "L1 +++$+++ C1 +++$+++ M1 +++$+++ A +++$+++ Hello there.\n"
            "L2 +++$+++ C2 +++$+++ M1 +++$+++ B +++$+++ General Kenobi.\n",
            encoding="utf-8",
        )
        (self.root / "movie_conversations.txt").write_text(
            "C1 +++$+++ C2 +++$+++ M1 +++$+++ ['L1', 'L2']\n",
            encoding="utf-8",
        )
        (self.root / "movie_titles_metadata.txt").write_text(
            "M1 +++$+++ title +++$+++ 1999 +++$+++ 8.0 +++$+++ 1000\n",
            encoding="utf-8",
        )
        self._write_csv(
            "personality.csv",
            ["Persona", "chat"],
            [{"Persona": "friendly", "chat": "How are you?\nI am well."}],
        )
        serialized = "['first turn' 'second turn' 'third turn']"
        for split in ("train", "validation", "test"):
            self._write_csv(
                f"{split}.csv",
                ["dialog", "act", "emotion"],
                [{"dialog": serialized, "act": "[]", "emotion": "[]"}],
            )

        worker = Ingest(self.root)
        documents = list(worker.ingest())
        self.assertEqual(worker.metrics.errors, 0)
        self.assertTrue(any(doc["type"] == "cornell_movies" for doc in documents))
        self.assertTrue(any(doc["type"] == "personachat" for doc in documents))
        self.assertEqual(
            {doc["split"] for doc in documents if doc["type"] == "dailydialog"},
            {"train", "validation", "test"},
        )
        self.assertFalse(any("title" in doc["text"] for doc in documents))
        self.assertTrue(all(doc["text"].strip() for doc in documents))
        self.assertTrue(all(doc["text"].endswith("<eos>") for doc in documents))

    def test_generic_text_still_supported_for_custom_inputs(self):
        sample = self.root / "notes.txt"
        sample.write_text("plain text", encoding="utf-8")
        documents = list(Ingest(self.root).ingest())
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["text"], "plain text")


if __name__ == "__main__":
    unittest.main()
