"""Safe, schema-aware conversational dataset ingestion."""

from __future__ import annotations

import ast
import io
import os
import tokenize
from pathlib import Path
from typing import Iterable

import pandas as pd
import pypdf
from charset_normalizer import from_path

from .JsonFormatter import log
from .Metrics import Metrics

MAX_FILE_BYTES = 10 * 1024 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
CSV_CHUNKSIZE = 500

USER_TOKEN = "<user>"
ASSISTANT_TOKEN = "<assistant>"
EOS_TOKEN = "<eos>"

IGNORED_DATASET_FILES = {
    ".ds_store",
    "movie_characters_metadata.txt",
    "movie_titles_metadata.txt",
    "raw_script_urls.txt",
    "readme.txt",
    "chameleons.pdf",
}


def format_conversation(turns: Iterable[str]) -> str:
    """Format alternating turns using the model's canonical control tokens."""
    cleaned = [str(turn).strip() for turn in turns if str(turn).strip()]
    if not cleaned:
        return ""
    pieces: list[str] = []
    for index, turn in enumerate(cleaned):
        role = USER_TOKEN if index % 2 == 0 else ASSISTANT_TOKEN
        pieces.extend((role, " ", turn, "\n"))
    pieces.append(EOS_TOKEN)
    return "".join(pieces)


def parse_serialized_turns(value: str) -> list[str]:
    """Parse the adjacent Python string literals used by the dialogue CSV files."""
    source = str(value).strip()
    if not source:
        return []

    turns: list[str] = []
    try:
        for token_info in tokenize.generate_tokens(io.StringIO(source).readline):
            if token_info.type == tokenize.STRING:
                parsed = ast.literal_eval(token_info.string)
                if isinstance(parsed, str) and parsed.strip():
                    turns.append(parsed.strip())
    except (SyntaxError, tokenize.TokenError, ValueError):
        turns = []

    if turns:
        return turns

    try:
        parsed = ast.literal_eval(source)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(turn).strip() for turn in parsed if str(turn).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


class Ingest:
    def __init__(
        self,
        path,
        *,
        max_file_bytes: int = MAX_FILE_BYTES,
        chunk_size: int = CHUNK_SIZE,
        csv_chunksize: int = CSV_CHUNKSIZE,
        allowlist_dirs: list[str] | None = None,
    ):
        self.path = Path(path).resolve()
        self.max_file_bytes = max_file_bytes
        self.chunk_size = chunk_size
        self.csv_chunksize = csv_chunksize
        self.allowlist_dirs = set(allowlist_dirs) if allowlist_dirs else None
        self.metrics = Metrics()
        self._seen_inodes: set[int] = set()

    # Safety gates -----------------------------------------------------

    def _is_dotfile(self, path: Path) -> bool:
        try:
            relative = path.resolve().relative_to(self.path)
        except ValueError:
            return True
        return any(part.startswith(".") for part in relative.parts)

    def _is_symlink(self, path: Path) -> bool:
        try:
            current = path
            while True:
                if current.is_symlink():
                    return True
                if current == self.path or current.parent == current:
                    break
                current = current.parent
        except OSError:
            return True
        return False

    def _is_confined(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.path)
            return True
        except ValueError:
            return False

    def _is_duplicate_inode(self, path: Path) -> bool:
        try:
            inode = path.stat().st_ino
            if not inode:
                return False
            if inode in self._seen_inodes:
                return True
            self._seen_inodes.add(inode)
        except OSError:
            return False
        return False

    def _safe(self, path: Path) -> bool:
        context = {"file": str(path)}
        if self._is_dotfile(path):
            self.metrics.skipped_dotfile += 1
            return False
        if self._is_symlink(path):
            self.metrics.skipped_symlink += 1
            log.warning("skip symlink", extra=context)
            return False
        if not self._is_confined(path):
            self.metrics.skipped_confined += 1
            log.warning("skip path outside dataset root", extra=context)
            return False
        if self._is_duplicate_inode(path):
            self.metrics.skipped_inode += 1
            return False
        try:
            size = path.stat().st_size
        except OSError:
            self.metrics.errors += 1
            log.exception("cannot stat input", extra=context)
            return False
        if size > self.max_file_bytes:
            self.metrics.skipped_size += 1
            log.warning("skip oversized file", extra={**context, "size": size})
            return False
        return True

    # Readers ----------------------------------------------------------

    def _read_text(self, path: Path) -> tuple[str, str]:
        result = from_path(path).best()
        if result is not None:
            return str(result), result.encoding or "unknown"
        self.metrics.encoding_warns += 1
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(), "utf-8-replace"

    def _emit(self, path: Path, doc_type: str, text: str, *, split: str = "train", **extra):
        if not text.strip():
            return None
        self.metrics.record(doc_type)
        return {
            "file": str(path),
            "type": doc_type,
            "split": split,
            "text": text,
            **extra,
        }

    def ingest_txt(self, path: Path):
        try:
            full_text, encoding = self._read_text(path)
            for offset in range(0, len(full_text), self.chunk_size):
                packet = self._emit(
                    path,
                    ".txt",
                    full_text[offset:offset + self.chunk_size],
                    encoding=encoding,
                )
                if packet:
                    yield packet
        except Exception:
            self.metrics.errors += 1
            log.exception("txt ingest failed", extra={"file": str(path)})

    def ingest_csv(self, path: Path):
        try:
            for frame in pd.read_csv(
                path,
                chunksize=self.csv_chunksize,
                dtype=str,
                keep_default_na=False,
            ):
                for row in frame.itertuples(index=False):
                    packet = self._emit(path, ".csv", " ".join(str(value) for value in row))
                    if packet:
                        yield packet
        except Exception:
            self.metrics.errors += 1
            log.exception("csv ingest failed", extra={"file": str(path)})

    def ingest_pdf(self, path: Path):
        try:
            with path.open("rb") as handle:
                reader = pypdf.PdfReader(handle)
                for page_number, page in enumerate(reader.pages, start=1):
                    packet = self._emit(path, ".pdf", page.extract_text() or "", page=page_number)
                    if packet:
                        yield packet
        except Exception:
            self.metrics.errors += 1
            log.exception("pdf ingest failed", extra={"file": str(path)})

    # Conversational schemas ------------------------------------------

    def ingest_dialog_csv(self, path: Path):
        split = path.stem.lower()
        try:
            for frame in pd.read_csv(
                path,
                chunksize=self.csv_chunksize,
                dtype=str,
                keep_default_na=False,
            ):
                if "dialog" not in frame.columns:
                    raise ValueError(f"{path.name} has no 'dialog' column")
                for value in frame["dialog"]:
                    turns = parse_serialized_turns(value)
                    text = format_conversation(turns)
                    packet = self._emit(path, "dailydialog", text, split=split, turns=turns)
                    if packet:
                        yield packet
        except Exception:
            self.metrics.errors += 1
            log.exception("dialogue CSV ingest failed", extra={"file": str(path)})

    def ingest_personality_csv(self, path: Path):
        try:
            for frame in pd.read_csv(
                path,
                chunksize=self.csv_chunksize,
                dtype=str,
                keep_default_na=False,
            ):
                if "chat" not in frame.columns:
                    raise ValueError(f"{path.name} has no 'chat' column")
                for _, row in frame.iterrows():
                    turns = [turn.strip() for turn in str(row["chat"]).splitlines() if turn.strip()]
                    text = format_conversation(turns)
                    packet = self._emit(
                        path,
                        "personachat",
                        text,
                        turns=turns,
                        persona=str(row.get("Persona", "")).strip(),
                    )
                    if packet:
                        yield packet
        except Exception:
            self.metrics.errors += 1
            log.exception("personality CSV ingest failed", extra={"file": str(path)})

    def _load_cornell_lines(self, path: Path) -> dict[str, str]:
        full_text, _ = self._read_text(path)
        line_map: dict[str, str] = {}
        for row in full_text.splitlines():
            fields = [field.strip() for field in row.split("+++$+++")]
            if len(fields) >= 5 and fields[0] and fields[4]:
                line_map[fields[0]] = fields[4]
        return line_map

    def ingest_cornell_conversations(self, path: Path):
        line_path = path.with_name("movie_lines.txt")
        try:
            if not line_path.exists():
                raise FileNotFoundError(f"Missing Cornell line file: {line_path}")
            line_map = self._load_cornell_lines(line_path)
            full_text, _ = self._read_text(path)
            for row in full_text.splitlines():
                fields = [field.strip() for field in row.split("+++$+++")]
                if len(fields) < 4:
                    continue
                try:
                    line_ids = ast.literal_eval(fields[-1])
                except (SyntaxError, ValueError):
                    continue
                turns = [line_map[line_id] for line_id in line_ids if line_id in line_map]
                packet = self._emit(
                    path,
                    "cornell_movies",
                    format_conversation(turns),
                    turns=turns,
                    line_ids=line_ids,
                )
                if packet:
                    yield packet
        except Exception:
            self.metrics.errors += 1
            log.exception("Cornell conversation ingest failed", extra={"file": str(path)})

    def ingest_cornell_lines(self, path: Path):
        """Fallback for a standalone movie_lines fixture without conversation indexes."""
        try:
            for line_id, text in self._load_cornell_lines(path).items():
                packet = self._emit(
                    path,
                    "cornell_movies",
                    format_conversation([text]),
                    turns=[text],
                    line_id=line_id,
                )
                if packet:
                    yield packet
        except Exception:
            self.metrics.errors += 1
            log.exception("Cornell line ingest failed", extra={"file": str(path)})

    # Routing and traversal -------------------------------------------

    def _detect_schema(self, path: Path):
        name = path.name.lower()
        if name in IGNORED_DATASET_FILES:
            return None
        if name == "movie_conversations.txt":
            return self.ingest_cornell_conversations
        if name == "movie_lines.txt":
            conversation_path = path.with_name("movie_conversations.txt")
            return None if conversation_path.exists() else self.ingest_cornell_lines
        if name == "personality.csv":
            return self.ingest_personality_csv
        if name in {"train.csv", "validation.csv", "test.csv"}:
            return self.ingest_dialog_csv
        if "dailydialog" in name:
            return self.ingest_txt
        return {
            ".txt": self.ingest_txt,
            ".csv": self.ingest_csv,
            ".pdf": self.ingest_pdf,
        }.get(path.suffix.lower())

    def ingest(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.path}")

        for directory, dirnames, filenames in os.walk(self.path, followlinks=False):
            directory_path = Path(directory)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not dirname.startswith(".")
                and (self.allowlist_dirs is None or dirname in self.allowlist_dirs)
                and self._is_confined(directory_path / dirname)
            ]

            for filename in sorted(filenames):
                path = directory_path / filename
                if not self._safe(path):
                    continue
                handler = self._detect_schema(path)
                if handler is not None:
                    yield from handler(path)

        log.info("ingest complete", extra={"metrics": self.metrics.report()})
