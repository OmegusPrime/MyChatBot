# MyChatBot

MyChatBot is a small decoder-only PyTorch chatbot trained from the conversational files in `Dataset/`. The terminal and Tkinter interfaces share the same tokenizer, checkpoint, generation engine, and bounded conversation history.

## Requirements

- Python 3.11 or 3.12
- PyTorch-compatible CPU or CUDA environment

Create and activate a virtual environment, then install the project dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.12 is installed under a different command, use that interpreter instead of `py -3.12`.

## Workflow

Generated files are stored under `artifacts/` and are ignored by Git.

### 1. Prepare the tokenizer and token streams

```powershell
python main_and_eval.py --prepare
```

Preparation uses explicit conversational schemas:

- `movie_lines.txt` and `movie_conversations.txt` are combined into Cornell conversations.
- `personality.csv` is parsed as alternating chat turns.
- `train.csv`, `validation.csv`, and `test.csv` retain their dataset split.
- Movie metadata, URLs, README text, and the unrelated PDF are excluded.

Preparation also builds `artifacts/response_pairs.db`, a searchable index of real
prompt-response pairs. To build or refresh only that index without retraining the
tokenizer or model:

```powershell
python main_and_eval.py --index
```

To intentionally replace current prepared artifacts:

```powershell
python main_and_eval.py --rebuild
```

### 2. Train the transformer

```powershell
python main_and_eval.py --train
```

Useful development options:

```powershell
python main_and_eval.py --train --epochs 1 --max-samples 5000 --device cpu
```

Training automatically prepares missing or stale data artifacts. It saves the best validation checkpoint, including model configuration and tokenizer compatibility metadata.

### 3. Start terminal chat

```powershell
python main_and_eval.py --chat
```

Type `RESET` to clear conversation history and `END` to exit.

### 4. Start the desktop GUI

```powershell
python chat_gui.py
```

The GUI requires the same `artifacts/tokenizer.json` and `artifacts/chatbot_transformer.pt` created by preparation and training. It does not require MySQL.

Responses use a confidence-based hybrid strategy:

1. Clear common intents such as greetings and thanks receive deterministic responses.
2. High-confidence matches use relevant replies from the conversational response index.
3. Low-confidence messages request clarification instead of emitting unrelated sampled text.

The small generative transformer remains available internally, but random model text is
not used as the default fallback.

## Tests

Run the complete suite:

```powershell
python -m pytest -q
```

The tests cover:

- Explicit conversational ingestion and dataset split handling
- Exact tokenizer round-trips for whitespace and Unicode
- Special-token encoding and removal
- Artifact preparation and token-ID bounds
- Transformer input invariants
- Autoregressive EOS behavior

## Architecture

```text
Dataset files
    ↓
Pipeline.Ingest
    ↓ role-formatted conversations
Pipeline.Tokenizer
    ↓ fixed token IDs
Model.TransformerCoreStack
    ↓ logits
Model.ChatBotInferenceEngine
    ↓ newly generated IDs
ChatService
    ├── terminal chat
    └── Tkinter GUI
```

The vocabulary is frozen before model training. New words are represented through byte-level BPE tokens; vocabulary rows are never added dynamically after the transformer is constructed.

## Troubleshooting

### Prepared artifacts are missing or stale

Run:

```powershell
python main_and_eval.py --prepare
```

If the dataset or tokenizer format was intentionally changed, use `--rebuild`.

### Checkpoint and tokenizer do not match

Rebuild and retrain together:

```powershell
python main_and_eval.py --rebuild --train
```

### Output is readable but not coherent

Confirm that training and validation loss decrease. Increase training epochs or sampled windows only after the ingestion and tokenizer tests pass. Sampling settings cannot compensate for an untrained checkpoint or malformed conversation data.

### CUDA is unavailable

Use CPU explicitly:

```powershell
python main_and_eval.py --train --device cpu
python main_and_eval.py --chat --device cpu
```

## Detailed remediation record

See [PROJECT_FIX_PLAN.md](PROJECT_FIX_PLAN.md) for the complete diagnosis, rationale, and acceptance criteria that guided the repair.
