"""Prepare data, train MyChatBot, or start terminal chat.

Examples:
    python main_and_eval.py --prepare
    python main_and_eval.py --train
    python main_and_eval.py --chat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from Pipeline.tokenizer import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "Dataset"
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_FORMAT_VERSION = 1

EMBED_DIM = 128
NUM_HEADS = 4
D_FF = 512
NUM_BLOCKS = 4
MAX_SEQ_LEN = 512
CONTEXT_LENGTH = 32
BATCH_SIZE = 64
LM_EPOCHS = 5
LEARNING_RATE = 5e-4
VOCAB_SIZE = 2500


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    tokenizer: Path
    train_stream: Path
    validation_stream: Path
    test_stream: Path
    metadata: Path
    checkpoint: Path
    response_index: Path


def artifact_paths(root: str | Path = DEFAULT_ARTIFACT_DIR) -> ArtifactPaths:
    root_path = Path(root).resolve()
    return ArtifactPaths(
        root=root_path,
        tokenizer=root_path / "tokenizer.json",
        train_stream=root_path / "train_token_stream.npy",
        validation_stream=root_path / "validation_token_stream.npy",
        test_stream=root_path / "test_token_stream.npy",
        metadata=root_path / "token_stream.meta.json",
        checkpoint=root_path / "chatbot_transformer.pt",
        response_index=root_path / "response_pairs.db",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(data_dir: str | Path) -> str:
    root = Path(data_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_numpy(path: Path, values: np.ndarray) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def _validate_stream(stream: np.ndarray, tokenizer: Tokenizer, name: str, *, required: bool) -> None:
    import numpy as np

    if stream.ndim != 1:
        raise ValueError(f"{name} token stream must be one-dimensional")
    if not np.issubdtype(stream.dtype, np.integer):
        raise TypeError(f"{name} token stream must use an integer dtype")
    if required and len(stream) <= CONTEXT_LENGTH + 1:
        raise ValueError(f"{name} token stream is too short: {len(stream)} tokens")
    if len(stream) and (int(stream.min()) < 0 or int(stream.max()) >= tokenizer.vocab_size):
        raise ValueError(f"{name} token stream contains IDs outside the tokenizer vocabulary")


def _metadata_is_current(paths: ArtifactPaths, data_dir: Path) -> bool:
    required = (
        paths.tokenizer,
        paths.train_stream,
        paths.validation_stream,
        paths.test_stream,
        paths.metadata,
    )
    if not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
        return (
            metadata.get("format_version") == ARTIFACT_FORMAT_VERSION
            and metadata.get("dataset_fingerprint") == dataset_fingerprint(data_dir)
            and metadata.get("tokenizer_sha256") == sha256_file(paths.tokenizer)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def prepare_artifacts(
    data_dir: str | Path = DEFAULT_DATASET_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    vocab_size: int = VOCAB_SIZE,
    force: bool = False,
) -> ArtifactPaths:
    """Build a tokenizer and split-specific token streams from clean conversations."""
    import numpy as np
    from Pipeline.ingest import Ingest
    from Pipeline.tokenizer import EOS_ID, Tokenizer

    data_path = Path(data_dir).resolve()
    paths = artifact_paths(artifact_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    if not force and _metadata_is_current(paths, data_path):
        if not paths.response_index.exists():
            from response_index import build_response_index

            print("Building missing conversational response index...")
            stats = build_response_index(data_path, paths.response_index)
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            metadata["response_index"] = stats
            _atomic_json(paths.metadata, metadata)
        print(f"Prepared artifacts are current in {paths.root}")
        return paths

    print("Preparing conversational dataset...")
    tokenizer = Tokenizer.train_from_ingest(
        data_dir=str(data_path),
        vocab_size=vocab_size,
        save_path=str(paths.tokenizer),
        split="train",
    )
    tokenizer_hash = sha256_file(paths.tokenizer)

    streams: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    record_counts = {"train": 0, "validation": 0, "test": 0}
    worker = Ingest(data_path)
    for packet in worker.ingest():
        split = packet.get("split", "train")
        if split not in streams:
            split = "train"
        encoded = tokenizer.encode(packet["text"], add_special=False)
        if not encoded:
            continue
        if encoded[-1] != EOS_ID:
            encoded.append(EOS_ID)
        streams[split].extend(encoded)
        record_counts[split] += 1

    if worker.metrics.errors:
        raise RuntimeError(
            f"Dataset preparation encountered {worker.metrics.errors} ingestion errors; "
            "inspect the preceding log messages"
        )

    arrays = {
        name: np.asarray(token_ids, dtype=np.int32)
        for name, token_ids in streams.items()
    }
    _validate_stream(arrays["train"], tokenizer, "train", required=True)
    _validate_stream(arrays["validation"], tokenizer, "validation", required=False)
    _validate_stream(arrays["test"], tokenizer, "test", required=False)

    _atomic_numpy(paths.train_stream, arrays["train"])
    _atomic_numpy(paths.validation_stream, arrays["validation"])
    _atomic_numpy(paths.test_stream, arrays["test"])

    from response_index import build_response_index

    print("Building conversational response index...")
    response_index_stats = build_response_index(data_path, paths.response_index)

    metadata = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "dataset_fingerprint": dataset_fingerprint(data_path),
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": tokenizer.vocab_size,
        "record_counts": record_counts,
        "token_counts": {name: int(len(values)) for name, values in arrays.items()},
        "ingestion_metrics": worker.metrics.report(),
        "response_index": response_index_stats,
    }
    _atomic_json(paths.metadata, metadata)

    print(f"Tokenizer vocabulary: {tokenizer.vocab_size:,}")
    for name in ("train", "validation", "test"):
        print(f"{name.title():>10}: {record_counts[name]:,} records, {len(arrays[name]):,} tokens")
    print(f"Prepared artifacts written to {paths.root}")
    return paths


def load_prepared_artifacts(
    data_dir: str | Path = DEFAULT_DATASET_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
):
    import numpy as np
    from Pipeline.tokenizer import Tokenizer

    data_path = Path(data_dir).resolve()
    paths = artifact_paths(artifact_dir)
    if not _metadata_is_current(paths, data_path):
        raise RuntimeError(
            "Prepared artifacts are missing or stale. Run main_and_eval.py --prepare first."
        )
    tokenizer = Tokenizer.from_pretrained(str(paths.tokenizer))
    train_stream = np.load(paths.train_stream, allow_pickle=False)
    validation_stream = np.load(paths.validation_stream, allow_pickle=False)
    test_stream = np.load(paths.test_stream, allow_pickle=False)
    _validate_stream(train_stream, tokenizer, "train", required=True)
    _validate_stream(validation_stream, tokenizer, "validation", required=False)
    _validate_stream(test_stream, tokenizer, "test", required=False)
    return paths, tokenizer, train_stream, validation_stream, test_stream


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _batches(
    stream: np.ndarray,
    *,
    context_len: int,
    batch_size: int,
    sample_count: int,
    rng: np.random.Generator,
    device,
):
    import numpy as np
    import torch

    max_start = len(stream) - context_len - 1
    if max_start < 0:
        raise ValueError(
            f"Token stream has {len(stream)} tokens but needs at least {context_len + 1}"
        )
    starts = rng.integers(0, max_start + 1, size=sample_count)
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        x_rows = [stream[start:start + context_len] for start in batch_starts]
        y_rows = [stream[start + 1:start + context_len + 1] for start in batch_starts]
        X = torch.as_tensor(np.asarray(x_rows), dtype=torch.long, device=device)
        Y = torch.as_tensor(np.asarray(y_rows), dtype=torch.long, device=device)
        yield X, Y


def evaluate_model(
    model,
    stream: np.ndarray,
    *,
    context_len: int,
    batch_size: int,
    device,
    seed: int,
    max_batches: int = 50,
) -> float:
    import numpy as np
    import torch
    import torch.nn as nn

    if len(stream) <= context_len + 1:
        return math.nan
    criterion = nn.CrossEntropyLoss()
    sample_count = min(max_batches * batch_size, len(stream) - context_len)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for X, Y in _batches(
            stream,
            context_len=context_len,
            batch_size=batch_size,
            sample_count=sample_count,
            rng=rng,
            device=device,
        ):
            logits = model(X)
            loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


def _save_checkpoint(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_generative_model(
    model,
    train_stream: np.ndarray,
    validation_stream: np.ndarray,
    tokenizer: Tokenizer,
    checkpoint_path: str | Path,
    *,
    epochs: int = LM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    context_len: int = CONTEXT_LENGTH,
    learning_rate: float = LEARNING_RATE,
    max_samples_per_epoch: int = 50_000,
    device,
    seed: int = 42,
    tokenizer_hash: str,
) -> Path:
    import numpy as np
    import torch
    import torch.nn as nn

    if epochs <= 0 or batch_size <= 0 or max_samples_per_epoch <= 0:
        raise ValueError("epochs, batch_size, and max_samples_per_epoch must be positive")
    if len(train_stream) <= context_len + 1:
        raise ValueError("Training token stream is too short")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sample_count = min(max_samples_per_epoch, len(train_stream) - context_len)
    checkpoint = Path(checkpoint_path).resolve()
    best_metric = math.inf

    print(f"Training on {device} with {sample_count:,} sampled windows per epoch")
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        total_loss = 0.0
        steps = 0
        for X, Y in _batches(
            train_stream,
            context_len=context_len,
            batch_size=batch_size,
            sample_count=sample_count,
            rng=rng,
            device=device,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}, step {steps + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        scheduler.step()
        train_loss = total_loss / max(steps, 1)
        validation_loss = evaluate_model(
            model,
            validation_stream,
            context_len=context_len,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        selection_metric = validation_loss if math.isfinite(validation_loss) else train_loss
        print(
            f"Epoch {epoch:02d}/{epochs:02d} - "
            f"train loss {train_loss:.4f} - "
            f"validation loss {validation_loss:.4f}"
        )

        if selection_metric < best_metric:
            best_metric = selection_metric
            _save_checkpoint(
                checkpoint,
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "model_config": model.config(),
                    "context_length": context_len,
                    "tokenizer_sha256": tokenizer_hash,
                    "vocab_size": tokenizer.vocab_size,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
            )
            print(f"Saved best checkpoint to {checkpoint}")

    return checkpoint


def train_command(args) -> Path:
    try:
        paths, tokenizer, train_stream, validation_stream, _ = load_prepared_artifacts(
            args.data_dir,
            args.artifact_dir,
        )
    except RuntimeError:
        paths = prepare_artifacts(
            args.data_dir,
            args.artifact_dir,
            vocab_size=args.vocab_size,
            force=args.rebuild,
        )
        paths, tokenizer, train_stream, validation_stream, _ = load_prepared_artifacts(
            args.data_dir,
            args.artifact_dir,
        )

    import torch
    from Model.Transformer import TransformerCoreStack

    device = _resolve_device(args.device)
    model = TransformerCoreStack(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_blocks=NUM_BLOCKS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)
    return train_generative_model(
        model,
        train_stream,
        validation_stream,
        tokenizer,
        paths.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_len=args.context_length,
        learning_rate=args.learning_rate,
        max_samples_per_epoch=args.max_samples,
        device=device,
        seed=args.seed,
        tokenizer_hash=sha256_file(paths.tokenizer),
    )


def run_terminal_chat(
    artifact_dir: str | Path, device: str = "auto", backend: str = "local"
) -> int:
    from chat_cli import run_terminal_chat as run_chat

    return run_chat(artifact_dir, device, backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare, train, or run MyChatBot")
    parser.add_argument("--prepare", action="store_true", help="build tokenizer and token streams")
    parser.add_argument("--train", action="store_true", help="train and save the transformer")
    parser.add_argument("--chat", action="store_true", help="start terminal chat")
    parser.add_argument("--index", action="store_true", help="build only the response-pair index")
    parser.add_argument("--rebuild", action="store_true", help="force artifact preparation")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--epochs", type=int, default=LM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument(
        "--backend", choices=("local", "legacy"), default="local",
        help="chat engine; local uses the instruction model, legacy uses your training checkpoint",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.prepare or args.train or args.chat or args.rebuild or args.index):
        parser.print_help()
        return 0

    if args.prepare or args.rebuild:
        prepare_artifacts(
            args.data_dir,
            args.artifact_dir,
            vocab_size=args.vocab_size,
            force=args.rebuild,
        )
    if args.index and not (args.prepare or args.rebuild):
        from response_index import build_response_index

        paths = artifact_paths(args.artifact_dir)
        paths.root.mkdir(parents=True, exist_ok=True)
        stats = build_response_index(args.data_dir, paths.response_index)
        if paths.metadata.exists():
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            metadata["response_index"] = stats
            _atomic_json(paths.metadata, metadata)
        print(f"Response index written to {paths.response_index}: {stats}")
    if args.train:
        train_command(args)
    if args.chat:
        return run_terminal_chat(args.artifact_dir, args.device, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
