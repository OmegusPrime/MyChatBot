"""SQLite FTS response-pair index for confidence-based conversational retrieval."""

from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

from Pipeline.ingest import Ingest

INDEX_FORMAT_VERSION = 1
WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
    "that", "the", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
}


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(str(text).casefold())


def normalize_prompt(text: str) -> str:
    return " ".join(words(text))


def _quality(response: str) -> float:
    length = len(response.strip())
    if length < 2 or length > 400:
        return -1.0
    word_count = len(words(response))
    if word_count == 0:
        return -1.0
    score = 1.0
    if 3 <= word_count <= 35:
        score += 0.5
    if response.rstrip().endswith((".", "!", "?")):
        score += 0.2
    score -= min(abs(length - 80) / 400, 0.5)
    return score


def build_response_index(data_dir: str | Path, destination: str | Path) -> dict:
    """Build an atomic response index from alternating user/assistant turns."""
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE response_pairs (
                id INTEGER PRIMARY KEY,
                normalized_prompt TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                quality REAL NOT NULL,
                UNIQUE(normalized_prompt, response)
            );
            CREATE INDEX response_pairs_normalized ON response_pairs(normalized_prompt);
            CREATE VIRTUAL TABLE response_fts USING fts5(
                prompt,
                content='response_pairs',
                content_rowid='id',
                tokenize='porter unicode61'
            );
            """
        )
        connection.execute(
            "INSERT INTO index_metadata(key, value) VALUES('format_version', ?)",
            (str(INDEX_FORMAT_VERSION),),
        )

        worker = Ingest(data_dir)
        batch: list[tuple[str, str, str, float]] = []
        source_pairs = 0
        for packet in worker.ingest():
            if packet.get("split", "train") != "train":
                continue
            turns = packet.get("turns") or []
            for index in range(0, len(turns) - 1, 2):
                prompt = str(turns[index]).strip()
                response = str(turns[index + 1]).strip()
                normalized = normalize_prompt(prompt)
                quality = _quality(response)
                if not normalized or quality < 0 or "http://" in response or "https://" in response:
                    continue
                batch.append((normalized, prompt, response, quality))
                source_pairs += 1
                if len(batch) >= 2000:
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO response_pairs
                            (normalized_prompt, prompt, response, quality)
                        VALUES (?, ?, ?, ?)
                        """,
                        batch,
                    )
                    connection.commit()
                    batch.clear()
        if batch:
            connection.executemany(
                """
                INSERT OR IGNORE INTO response_pairs
                    (normalized_prompt, prompt, response, quality)
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )
        if worker.metrics.errors:
            raise RuntimeError(f"Cannot build response index: {worker.metrics.errors} ingestion errors")

        connection.execute("INSERT INTO response_fts(response_fts) VALUES('rebuild')")
        unique_pairs = int(connection.execute("SELECT COUNT(*) FROM response_pairs").fetchone()[0])
        connection.execute(
            "INSERT INTO index_metadata(key, value) VALUES('source_pairs', ?)",
            (str(source_pairs),),
        )
        connection.execute(
            "INSERT INTO index_metadata(key, value) VALUES('unique_pairs', ?)",
            (str(unique_pairs),),
        )
        connection.commit()
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()
        os.replace(temporary, target)
        return {"source_pairs": source_pairs, "unique_pairs": unique_pairs}


class ResponseRetriever:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Response index does not exist: {self.path}")
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        version = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key='format_version'"
        ).fetchone()
        if version is None or int(version[0]) != INDEX_FORMAT_VERSION:
            self.connection.close()
            raise RuntimeError("Response index format is incompatible; rebuild the index")

    def close(self) -> None:
        self.connection.close()

    def retrieve(self, query: str, *, threshold: float = 0.72) -> tuple[str, float] | None:
        normalized = normalize_prompt(query)
        if not normalized:
            return None

        exact = self.connection.execute(
            """
            SELECT response, quality
            FROM response_pairs
            WHERE normalized_prompt = ?
            ORDER BY quality DESC, length(response) ASC, response ASC
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
        if exact:
            return str(exact[0]), 1.0

        query_words = words(query)
        content_words = [word for word in query_words if word not in STOP_WORDS]
        if len(set(content_words)) < 2:
            return None

        fts_terms = list(dict.fromkeys(content_words))[:12]
        match_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in fts_terms)
        candidates = self.connection.execute(
            """
            SELECT pairs.prompt, pairs.response, pairs.quality
            FROM response_fts
            JOIN response_pairs AS pairs ON pairs.id = response_fts.rowid
            WHERE response_fts MATCH ?
            ORDER BY bm25(response_fts)
            LIMIT 50
            """,
            (match_query,),
        ).fetchall()

        query_counter = Counter(content_words)
        query_set = set(query_counter)
        best: tuple[str, float, float] | None = None
        for prompt, response, quality in candidates:
            candidate_words = [word for word in words(prompt) if word not in STOP_WORDS]
            candidate_set = set(candidate_words)
            if not candidate_set:
                continue
            overlap = query_set & candidate_set
            coverage = len(overlap) / len(query_set)
            jaccard = len(overlap) / len(query_set | candidate_set)
            similarity = 0.75 * coverage + 0.25 * jaccard
            if best is None or (similarity, quality) > (best[1], best[2]):
                best = (str(response), similarity, float(quality))

        if best and best[1] >= threshold:
            return best[0], best[1]
        return None


__all__ = [
    "INDEX_FORMAT_VERSION",
    "ResponseRetriever",
    "build_response_index",
    "normalize_prompt",
    "words",
]
