# MyChatBot Complete Remediation Plan

> **Implementation status (2026-08-22):** The source-code remediation in this plan has been implemented. The project now has a local Python 3.12 environment, declared dependencies, schema-aware ingestion, exact tokenizer round-trips, versioned preparation artifacts, restored training/checkpoint logic, a shared terminal/GUI backend, and automated coverage. The repaired suite passes 16 tests plus 8 parameterized tokenizer cases. The real dataset produces 105,136 conversational records with zero ingestion errors, and 239,690 unique prompt-response pairs are indexed for confidence-based retrieval. A full production checkpoint is intentionally not committed; run the preparation and training commands in `README.md` to create machine-local artifacts.

## 1. Objective

The immediate objective is to make the project reliably accept a prompt and return a readable response. In this document, **readable** means:

- The application starts without syntax, import, database, or checkpoint errors.
- Text survives tokenizer encode/decode without artifacts such as `Ġ`, `�`, or unexpected special tokens.
- The model uses the same fixed vocabulary during training and inference.
- A trained checkpoint is saved and loaded successfully.
- A prompt produces non-empty decoded text.

Readable does not automatically mean intelligent or factually useful. Coherent conversation additionally depends on clean conversational data, decreasing validation loss, adequate training, and sensible generation settings.

The fastest reliable route is:

```text
Python environment
        ↓
Package/import repair
        ↓
Clean conversational records
        ↓
Correct tokenizer round-trip
        ↓
Token-stream cache
        ↓
Model training and checkpoint
        ↓
Terminal response verification
        ↓
GUI integration
```

The terminal application should be treated as the reference implementation. The GUI should only wrap the working terminal backend; it must not maintain a separate tokenizer, vocabulary, embedding implementation, or model architecture.

---

## 2. Confirmed Problems and Their Effects

| Observed problem | Location | Effect |
|---|---|---|
| Invalid `self.engine.(` expression | `chat_gui.py:200` | The GUI cannot be parsed or started. |
| Import of nonexistent `LanguageModelHead` | `chat_gui.py:18` | The GUI fails during imports even after its syntax error is fixed. |
| GUI uses an obsolete `ChatBotInferenceEngine` constructor | `chat_gui.py:125-129` | The GUI cannot initialize the current inference engine. |
| GUI uses invalid generation argument names | `chat_gui.py:200-206` | Generation fails even if initialization is repaired. |
| Pipeline modules use top-level sibling imports | `Pipeline/ingest.py:4-5`, `Pipeline/tokenizer.py:5` | Running from the repository root raises `ModuleNotFoundError`. |
| `X` and `Y` batch tensors are never created | `main_and_eval.py:121-127` | Clean training raises `NameError` on the first epoch. |
| Current training function no longer saves a checkpoint | `main_and_eval.py` | Every clean launch attempts training again and no trained state persists. |
| SQLite insertion uses `%s` and only two placeholders for three columns | `main_and_eval.py:81-83` | Cache creation raises `sqlite3.OperationalError`. |
| `cornell_movies` is absent from the metrics dictionary | `Pipeline/Metrics.py:12-15` | Cornell ingestion raises `KeyError` and returns zero movie lines. |
| All filenames containing `movie` use the dialogue parser | `Pipeline/ingest.py:285-287` | Metadata and conversation-index files are misinterpreted as dialogue. |
| `personality.csv` is sent to the PersonaChat text parser | `Pipeline/ingest.py:283-284` | The file produces no useful conversational records. |
| Tokenizer encodes a manually inserted Unicode `Ġ` | `Pipeline/tokenizer.py:68-70` | `hello world` decodes as `helloĠworld`. |
| Decoder removes only four special tokens | `Pipeline/tokenizer.py:244-246` | Role/separator tokens can appear in user-visible responses. |
| Training and inference use different role-token spellings | `Pipeline/tokenizer.py:11-12`, `main_and_eval.py:147-149` | Inference role IDs are guessed rather than resolved from the trained tokenizer. |
| Tests import nonexistent top-level modules and a nonexistent `ingest` function | `test/Pipeline_test.py:5-6` | The test module cannot be imported. |
| Test dataset path is `/Dataset` | `test/Pipeline_test.py:8` | Tests look outside the repository on Windows. |
| GUI vocabulary grows after model construction | `chat_gui.py:65-97`, `chat_gui.py:187-193` | New token IDs exceed the fixed PyTorch embedding/output dimensions. |
| GUI creates a fresh random model on every start | `chat_gui.py:118-129` | Even a runnable GUI would produce untrained output. |
| GUI mixes NumPy and PyTorch model components | `chat_gui.py:15-19`, `chat_gui.py:121-129` | Components have incompatible APIs, tensors, and training state. |
| A database password is hard-coded | `chat_gui.py:22-27` | Security risk and environment-specific startup failure. |
| No dependency manifest or setup instructions exist | Repository root | The project cannot be reproduced reliably on another machine. |

---

## 3. Architecture Decision

Use one implementation throughout the project:

```text
Pipeline.Tokenizer
        ↓ token IDs
Model.TransformerCoreStack (PyTorch)
        ↓ logits
Model.ChatBotInferenceEngine
        ↓ generated IDs
Pipeline.Tokenizer.decode
        ↓ readable text
Terminal and GUI
```

The following components must not be part of the active chat path:

- `DatabaseBackedTokenizer`
- Dynamic MySQL vocabulary creation
- `Embedding/DenseEmbeddingEngine.py`
- `Embedding/TrainingSampler.py`
- NumPy attention and layer classes from `Model/Attention.py` and `Model/Layers.py`
- A separate GUI-only language-model head

These files can be moved to a `legacy/` directory temporarily if their history is useful. They should not be imported by the terminal or GUI application.

The PyTorch transformer's vocabulary is fixed when the model is created. Adding words to a database after training cannot expand only part of the model. Expanding a vocabulary requires coordinated resizing and retraining of both the input embedding and output projection. For this project, freezing the BPE tokenizer after training is the correct design.

---

## 4. Environment and Project Setup

### 4.1 Select a supported Python runtime

Use a real project environment based on Python 3.11 or 3.12. Confirm that the interpreter selected by the IDE is the same interpreter used in the terminal.

The current IDE references Python 3.13, but no working system Python installation was discoverable during diagnosis. Do not rely only on an IDE SDK label; verify the interpreter executable actually exists.

### 4.2 Add a dependency manifest

Create either `requirements.txt` or `pyproject.toml`. The minimum runtime/test dependencies are:

```text
numpy
torch
pandas
pypdf
charset-normalizer
pytest
```

`mysql-connector-python` is unnecessary after removing the MySQL-backed tokenizer. Pin exact versions only after the repaired project passes on the selected Python version.

### 4.3 Add setup and run instructions

Expand `README.md` with:

1. Supported Python version.
2. Environment creation.
3. Dependency installation.
4. Dataset preparation command.
5. Training command.
6. Terminal chat command.
7. GUI command.
8. Test command.
9. Expected artifact locations.

Prefer explicit commands such as:

```text
python -m pytest
python main_and_eval.py --prepare
python main_and_eval.py --train
python main_and_eval.py --chat
python chat_gui.py
```

The command-line flags do not exist yet; add them as described later in this plan.

### 4.4 Add repository ignore rules

Update `.gitignore` to exclude generated and secret material:

```text
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
*.db
*.npy
*.pt
.env
```

If trained checkpoints are intentionally versioned, store them through an artifact mechanism rather than committing large, changing binary files without documentation.

---

## 5. Repair the Python Package Structure

### 5.1 Add package markers

Add empty `__init__.py` files to:

```text
Pipeline/__init__.py
Model/__init__.py
Embedding/__init__.py
test/__init__.py
```

`Embedding` can later be removed from the active design, but making packages explicit avoids environment-dependent namespace behavior during the transition.

### 5.2 Correct internal imports

In `Pipeline/ingest.py`, replace:

```python
from Metrics import Metrics
from JsonFormatter import JsonFormatter, log
```

with:

```python
from .Metrics import Metrics
from .JsonFormatter import log
```

`JsonFormatter` itself is not used by `ingest.py`, so do not import it there.

In `Pipeline/tokenizer.py`, replace:

```python
from ingest import Ingest
```

with:

```python
from .ingest import Ingest
```

In tests, import public package objects:

```python
from Pipeline.ingest import Ingest
from Pipeline.tokenizer import Tokenizer
```

### 5.3 Stop modifying `sys.path` where possible

Once the repository is a valid package and applications are launched from the repository root, the manual `sys.path.append(BASE_DIR)` blocks should not be necessary. They can temporarily remain during repair, but the final project should use ordinary package imports consistently.

### 5.4 Import acceptance check

The following commands must complete without errors before proceeding:

```text
python -c "from Pipeline.ingest import Ingest"
python -c "from Pipeline.tokenizer import Tokenizer"
python -c "from Model.Transformer import TransformerCoreStack"
python -c "from Model.Generator import ChatBotInferenceEngine"
```

---

## 6. Repair Dataset Ingestion

### 6.1 Fix the metrics dictionary

Add `cornell_movies` and any other emitted schema names to `Metrics.by_type`:

```python
by_type: dict = field(default_factory=lambda: {
    ".txt": 0,
    ".csv": 0,
    ".pdf": 0,
    "dailydialog": 0,
    "personachat": 0,
    "cornell_movies": 0,
})
```

Every handler must increment an existing key. Add a test that enumerates all possible emitted `type` values and verifies that each exists in `by_type`.

### 6.2 Route files by exact schema, not broad substrings

Replace broad checks such as `if "movie" in name` with exact filename routing.

Recommended initial routing:

| Filename | Action |
|---|---|
| `movie_lines.txt` | Parse as Cornell dialogue lines. |
| `movie_conversations.txt` | Ignore initially, or use only to reconstruct ordered conversations after line-ID parsing is implemented. |
| `movie_characters_metadata.txt` | Ignore for language-model training. |
| `movie_titles_metadata.txt` | Ignore for language-model training. |
| `raw_script_urls.txt` | Ignore. |
| `README.txt` | Ignore. |
| `personality.csv` | Parse with a dedicated CSV handler. |
| `train.csv` | Parse its `dialog` column with a dedicated dialogue handler. |
| `validation.csv` | Parse its `dialog` column as validation data. |
| `test.csv` | Parse its `dialog` column as test data. |
| `chameleons.pdf` | Exclude from the first conversational model unless it is intentionally relevant. |

Start with an explicit allowlist. General document ingestion can be added after conversational training works.

### 6.3 Parse Cornell lines correctly

For `movie_lines.txt`:

1. Split on `+++$+++`.
2. Require at least five fields.
3. Use field index `4` as the dialogue text.
4. Strip surrounding whitespace.
5. Reject empty lines.

Do not apply this parser to the metadata files. Their field at index `4` is not dialogue.

For better conversational structure, later load `movie_conversations.txt`, resolve its line-ID sequences against `movie_lines.txt`, and alternate `<user>`/`<assistant>` roles. For the minimum viable model, individual movie lines may be accepted as text records, but they provide weaker conversational supervision.

### 6.4 Parse `personality.csv` as CSV

The file contains `Persona` and `chat` columns. A dedicated handler should:

1. Read rows using pandas.
2. Treat the persona text as optional context.
3. Split the `chat` field into non-empty turns.
4. Alternate user and assistant roles.
5. Emit one formatted conversation per row.

Example training text:

```text
<user> hi, how are you doing?
<assistant> i am getting ready to exercise.
<user> you must be very fast.
<assistant> hunting is one of my favorite hobbies.
<eos>
```

Persona context can later be represented with a `<persona>` token, but avoid adding another special-token format until the basic role format works.

### 6.5 Parse `train.csv`, `validation.csv`, and `test.csv`

The `dialog` column contains a serialized list of turns. Parse it safely with `ast.literal_eval`; never use `eval`.

For each row:

1. Parse `dialog` into a list of strings.
2. Discard malformed or empty rows with a logged warning.
3. Alternate `<user>` and `<assistant>` tokens.
4. End each conversation with `<eos>`.
5. Keep training, validation, and test splits separate.

The `act` and `emotion` columns should be ignored for the first language-model implementation unless the model explicitly learns those labels.

### 6.6 Emit conversation boundaries

Do not concatenate unrelated documents with no boundary. Every record added to the token stream must end with `<eos>`. Otherwise, the model learns that the end of one unrelated document naturally continues into another.

### 6.7 Make ingestion failures visible

The current handlers catch broad exceptions, log them, and continue. This is acceptable for large mixed-document ingestion only if the caller checks the final metrics.

After ingestion:

- Fail preparation if `metrics.ingested == 0`.
- Fail preparation if required data sources produced zero rows.
- Print a compact type-by-type count.
- Treat unexpected errors in required handlers as fatal.

### 6.8 Ingestion acceptance checks

Before tokenizer training:

- `movie_lines.txt` produces more than zero dialogue records.
- No movie metadata file produces dialogue records.
- `personality.csv` produces more than zero conversations.
- Training, validation, and test data remain distinguishable.
- All emitted records contain non-empty `text`, `file`, and `type` fields.
- All conversations end with `<eos>` or expose structured turns that the formatter converts to `<eos>`-terminated text.
- Required ingestion completes with zero unexpected errors.

---

## 7. Repair the Tokenizer

### 7.1 Fix the confirmed leading-space bug

The current `_pretokenize` method manually inserts the Unicode character `Ġ` and then UTF-8 byte-encodes that character. This double-encodes the marker. Decoding reconstructs a literal `Ġ` rather than an ASCII space.

Replace the manual marker construction:

```python
raw = ("Ġ" + word.lstrip(" ")) if word.startswith(" ") else word
chars = Tokenizer._word_to_bytes(raw)
```

with byte encoding of the original matched text:

```python
raw = m.group()
chars = Tokenizer._word_to_bytes(raw)
```

`BYTE_ENCODER` already maps byte `32` (space) to its visible byte-level representation. The code must not insert that representation before byte encoding.

### 7.2 Preserve text in `decode`

The decoder should:

1. Resolve IDs to tokens.
2. Remove all selected special tokens before byte reconstruction.
3. Join the remaining byte-level symbols.
4. Require every normal symbol to exist in `BYTE_DECODER`.
5. Convert symbols back to bytes.
6. Decode the complete byte array as UTF-8.

Do not silently drop characters with:

```python
if ch in BYTE_DECODER
```

Silently dropping unknown characters hides tokenizer corruption. Raise a descriptive error during tests or explicitly substitute an unknown marker.

Do not collapse whitespace with `re.sub(r"\s+", " ", ...)` inside the tokenizer. Exact tokenizer round-tripping should preserve the original bytes. If the chat interface wants display normalization, do that after decoding in the interface layer.

### 7.3 Standardize special tokens

Choose one spelling and use it everywhere. The simplest choice matches the existing tokenizer vocabulary:

```python
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<user>",
    "<assistant>",
    "<sep>",
    "<mask>",
]
```

Then change inference to use `<user>` and `<assistant>`, not `<|user|>` and `<|assistant|>`.

Do not use fallback IDs such as `vocab.get("<|user|>", 4)`. Resolve required IDs explicitly and fail with a clear error if a required token is missing:

```python
user_id = tokenizer.vocab["<user>"]
assistant_id = tokenizer.vocab["<assistant>"]
eos_id = tokenizer.vocab["<eos>"]
```

This prevents silently assigning a semantic role to the wrong vocabulary entry.

### 7.4 Skip every special token during user-visible decode

When `skip_special=True`, remove `set(SPECIAL_TOKENS)`, not only pad/unknown/begin/end tokens. This prevents `<user>`, `<assistant>`, `<sep>`, and `<mask>` from leaking into responses.

Decide deliberately whether `<unk>` should be removed or displayed. For debugging, displaying `<unk>` is useful. For the final UI, an unknown symbol can be omitted only after the tokenizer is proven to cover the input bytes.

### 7.5 Preserve tokenizer/model compatibility

Once model training begins, freeze the tokenizer. Save the following metadata next to the model checkpoint:

- Tokenizer file hash.
- Vocabulary size.
- Ordered special-token mapping.
- BPE merge count.
- Tokenizer format version.

Refuse to load a checkpoint if its tokenizer metadata does not match the active tokenizer.

### 7.6 Correct the streaming claim or implementation

`train_from_ingest` accepts a generator, but `Tokenizer.train` stores every pre-tokenized word in a `corpus` list. It is therefore not truly streaming. This is not the first readability blocker, but the comment and behavior should agree.

Choose one:

- Keep the current in-memory algorithm and document its memory requirements.
- Replace it with a bounded/frequency-based implementation suitable for large corpora.

For the current dataset, the first option is acceptable for the initial repair if memory use remains safe.

### 7.7 Tokenizer acceptance tests

Add parameterized tests for:

```text
hello world
 leading space
multiple   spaces
punctuation: hello, world!
newlines\nand\ttabs
café
नमस्ते
emoji 🙂
```

Required assertions:

```python
decoded = tokenizer.decode(tokenizer.encode(sample), skip_special=False)
assert decoded == sample
assert all(token_id in tokenizer.inv_vocab for token_id in tokenizer.encode(sample))
```

Also verify:

- Empty input encodes to an empty list when special tokens are disabled.
- Save/load preserves vocabulary and merges.
- Encoding is identical before and after save/load.
- User-visible decode contains no special-token strings.

Do not proceed to model training until these tests pass.

---

## 8. Repair Token-Stream Caching

### 8.1 Prefer a simpler cache path

SQLite is not necessary for the minimum working application. The simplest reliable preparation flow is:

```text
Ingest records
    ↓
Format conversation text
    ↓
Tokenizer.encode
    ↓
Append IDs and `<eos>`
    ↓
Save `token_stream.npy`
```

Removing SQLite eliminates an unnecessary failure point and avoids storing a second full copy of the training corpus.

### 8.2 If SQLite is retained, fix its SQL

Replace:

```python
"INSERT INTO processed_documents (file_name, doc_type, raw_text) VALUES (%s, %s)"
```

with:

```python
"INSERT INTO processed_documents (file_name, doc_type, raw_text) VALUES (?, ?, ?)"
```

Use a connection context manager so exceptions roll back safely and the connection always closes.

### 8.3 Version the cache

A cached token stream is valid only for the exact tokenizer and dataset used to create it. Store cache metadata containing:

- Tokenizer hash.
- Dataset file names, sizes, and modification times or content hashes.
- Formatting version.
- Special-token mapping.
- Context length if windows are precomputed.

If metadata differs, rebuild the cache instead of loading stale IDs.

### 8.4 Validate cache output

Before saving or training:

- Ensure the stream is one-dimensional.
- Ensure its dtype is an integer type.
- Ensure it is longer than `CONTEXT_LENGTH + 1`.
- Ensure its minimum ID is at least zero.
- Ensure its maximum ID is smaller than `tokenizer.vocab_size`.
- Report token count and source-record count.

Write generated artifacts atomically where practical so an interrupted preparation does not leave a valid-looking partial cache.

---

## 9. Restore and Harden Model Training

### 9.1 Restore batch construction

The current `train_generative_model` references undefined `X` and `Y`. Restore an actual batch loop:

```python
total_tokens = len(token_stream)
max_start = total_tokens - context_len - 1
if max_start <= 0:
    raise ValueError("Token stream is too short for the configured context length")

sample_count = min(50_000, max_start)

for epoch in range(1, epochs + 1):
    start_indices = np.random.randint(0, max_start, size=sample_count)
    total_loss = 0.0
    steps = 0

    for offset in range(0, len(start_indices), batch_size):
        batch_starts = start_indices[offset:offset + batch_size]
        if len(batch_starts) == 0:
            continue

        x_rows = [token_stream[s:s + context_len] for s in batch_starts]
        y_rows = [token_stream[s + 1:s + context_len + 1] for s in batch_starts]

        X = torch.as_tensor(np.asarray(x_rows), dtype=torch.long, device=device)
        Y = torch.as_tensor(np.asarray(y_rows), dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1
```

The scheduler should step once per epoch after the batch loop. The current `max_samples_per_epoch` variable must either control sampling or be removed.

### 9.2 Save a complete checkpoint

After successful training, save more than a raw state dictionary:

```python
torch.save(
    {
        "model_state": model.state_dict(),
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "d_ff": D_FF,
            "num_blocks": NUM_BLOCKS,
            "max_seq_len": 512,
        },
        "tokenizer_hash": tokenizer_hash,
        "epoch": epochs,
        "training_loss": mean_epoch_loss,
    },
    MODEL_CHECKPOINT_PATH,
)
```

On load:

1. Read configuration from the checkpoint.
2. Verify tokenizer hash and vocabulary size.
3. Construct the model using the saved configuration.
4. Load the state strictly.
5. Move the model to the selected device.
6. Call `model.eval()` before generation.

### 9.3 Separate preparation, training, and chat

Do not silently perform expensive tokenizer training and model training when the user only wants to chat. Add explicit modes:

```text
--prepare   Build tokenizer and token cache.
--train     Train and save a checkpoint.
--chat      Require existing compatible artifacts and start chat.
--rebuild   Explicitly replace existing generated artifacts.
```

If `--chat` cannot find compatible artifacts, print a clear command telling the user what to run. Do not start inference with random weights.

### 9.4 Use deterministic diagnostics

Set seeds during debugging:

```python
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

Determinism is useful for reproducing training failures. It is not required for final sampling behavior.

### 9.5 Add validation

Track validation loss separately from training loss. At minimum:

- Evaluate at the end of every epoch.
- Do not backpropagate during validation.
- Save the best validation checkpoint, not merely the final epoch.
- Stop if loss becomes non-finite.

Training loss should decrease on a small overfitting test. Validation loss is the better indicator of whether general response quality is improving.

### 9.6 Protect model invariants

In `Model/Transformer.py`, validate:

- `embed_dim % num_heads == 0`.
- Input tensor dtype is `torch.long`.
- Sequence length does not exceed `max_seq_len`.
- Token IDs fall within `[0, vocab_size)`.
- Vocabulary size is positive.

Store `max_seq_len` and `vocab_size` on the model instance so errors can be descriptive.

### 9.7 Training acceptance checks

Before full training:

1. Overfit a tiny corpus of a few short conversations.
2. Confirm loss decreases substantially.
3. Save the checkpoint.
4. Create a new model instance.
5. Reload the checkpoint.
6. Confirm the reloaded model produces the same logits for a fixed input.

Only then train on the full prepared corpus.

---

## 10. Repair Generation and Terminal Chat

### 10.1 Make the generation API unambiguous

`generate_response` currently returns the retained prompt plus generated tokens. Callers then calculate offsets, which becomes incorrect whenever the prompt was truncated.

Prefer returning only newly generated IDs:

```python
generated_ids = engine.generate_response(prompt_ids, ...)
response_text = tokenizer.decode(generated_ids, skip_special=True)
```

Internally maintain `working_sequence`, but accumulate generated IDs in a separate list and return that list.

### 10.2 Format prompts exactly like training data

For each turn:

```text
<user> user text <assistant>
```

Use exact token IDs from the tokenizer. Training records must use the same order and spelling. Append generated assistant tokens and `<eos>` to history as appropriate.

### 10.3 Stop on meaningful boundaries

Generation should stop when it produces:

- `<eos>`
- Optionally `<user>` if the model starts the next turn
- The configured maximum number of new tokens

Mask tokens that should never be sampled in an assistant response, such as `<pad>`, `<bos>`, and `<mask>`.

### 10.4 Improve basic readability controls

After correctness is established, add:

- A small repetition penalty.
- Optional no-repeat n-gram handling.
- A minimum response length before allowing `<eos>`.
- Safe temperature validation.
- Safe top-k and top-p validation.
- A retry if every sampled token is filtered or decoding is empty.

Start with conservative values:

```text
temperature: 0.8
top_k: 20
top_p: 0.9
max_new_tokens: 40
```

These values do not compensate for an untrained model or bad data. They only shape valid model output.

### 10.5 Handle empty output honestly

If decoding produces an empty string, do not silently present `...` as a successful model response. Report a diagnostic such as:

```text
The model produced only control tokens. Try a different prompt or inspect the checkpoint.
```

During development, log the generated token IDs and their token strings.

### 10.6 Terminal acceptance checks

For a fixed checkpoint and seed:

- Application starts without rebuilding artifacts.
- Prompt IDs are all within the vocabulary.
- At least one non-special token is generated.
- Decoded text contains no `Ġ`, `�`, raw role tags, or unexpected `<unk>` bursts.
- Conversation history never exceeds the configured context window passed to the model.
- `END` exits cleanly.

This is the milestone that satisfies the minimum readable-response goal.

---

## 11. Replace the GUI Backend

### 11.1 Remove incompatible imports and classes

Remove these GUI dependencies:

```python
import mysql.connector
from Model.Layers import ...
from Model.Attention import ...
from Model.Generator import LanguageModelHead
from Embedding.DenseEmbeddingEngine import DenseEmbeddingEngine
```

Remove `MYSQL_DB_CONFIG` and the entire `DatabaseBackedTokenizer` class.

### 11.2 Create one reusable chat service

Create a module such as `chat_backend.py` containing a `ChatService` that:

1. Loads the tokenizer.
2. Loads and validates the checkpoint.
3. Constructs `ChatBotInferenceEngine`.
4. Owns conversation history.
5. Exposes `reply(text: str) -> str`.

Both the terminal and GUI must call this service. This prevents the two entry points from drifting into incompatible architectures again.

Suggested boundary:

```python
class ChatService:
    def __init__(self, tokenizer_path, checkpoint_path, device=None):
        ...

    def reply(self, user_text: str) -> str:
        ...

    def reset(self) -> None:
        ...
```

### 11.3 Fix GUI generation calls

The invalid expression at `chat_gui.py:200` should disappear entirely. The GUI should simply call:

```python
response = self.chat_service.reply(prompt_text)
```

The backend, not the GUI, should manage token IDs, context truncation, sampling, and decoding.

### 11.4 Keep the UI responsive and thread-safe

- Run generation in a daemon worker thread.
- Disable the Send button while a response is being generated.
- Re-enable it after success or failure.
- Send all Tkinter widget updates through `root.after`.
- Prevent simultaneous model calls with a lock.
- Catch backend exceptions and display a concise message while logging the full traceback.
- Provide a Reset Conversation button that calls `ChatService.reset()`.

### 11.5 Load once

Load the tokenizer and checkpoint once during application initialization. Do not recreate random embeddings or transformer weights for each message or launch.

### 11.6 GUI acceptance checks

- GUI starts with no database dependency.
- It refuses to start if compatible trained artifacts are absent.
- The same prompt produces the same backend behavior as terminal chat when sampling is seeded.
- Multiple turns preserve bounded history.
- Sending several messages does not create new vocabulary IDs.
- The window remains responsive during generation.

---

## 12. Repair and Expand Tests

### 12.1 Fix the existing test module

Rename `test/Pipeline_test.py` to a conventional name such as:

```text
test/test_pipeline.py
```

Replace imports with package imports and instantiate `Ingest`, not a nonexistent `ingest` function.

Do not use `DATASET_DIR = r"/Dataset"`. For integration tests, calculate the repository-relative path:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
```

Unit tests should prefer temporary fixture directories rather than scanning the complete dataset in every `setUp` call.

### 12.2 Required test groups

#### Import tests

- Every active module imports from the repository root.
- GUI/backend imports do not require a running database.

#### Ingestion tests

- TXT, CSV, and PDF happy paths.
- Cornell line parsing.
- Metadata exclusion.
- Personality CSV parsing.
- Serialized dialogue parsing.
- Empty and malformed rows.
- Metrics keys and counts.
- Path confinement and symlink behavior.

#### Tokenizer tests

- Exact ASCII and Unicode round-trips.
- Whitespace preservation.
- Special-token insertion and removal.
- ID bounds.
- Save/load equivalence.
- Tokenizer hash stability.

#### Cache tests

- Creation and reload.
- Invalidation after tokenizer change.
- Invalidation after dataset change.
- Empty/short stream rejection.
- Token-ID bounds.

#### Model tests

- Forward-pass shape.
- Causal behavior.
- Invalid head/dimension rejection.
- Invalid token-ID rejection.
- Tiny-corpus overfit.
- Checkpoint round-trip.

#### Generation tests

- Returns newly generated IDs only.
- Stops at EOS.
- Does not sample masked control tokens.
- Respects context length.
- Produces decodable output.

#### GUI/backend tests

- `ChatService.reply` uses fixed tokenizer and model.
- History reset.
- History truncation.
- Missing-artifact error.

### 12.3 Test execution gate

Run all tests before full training:

```text
python -m pytest -q
```

Do not accept manual GUI output as a substitute for tokenizer, cache, and checkpoint tests.

---

## 13. Security and Configuration Fixes

### 13.1 Remove the hard-coded database credential

Delete the credential from `chat_gui.py`. If it is real, rotate it because it has been stored in source control.

The repaired architecture does not require MySQL. If a database is added later for chat history, load credentials from environment variables or a local `.env` file excluded from version control.

### 13.2 Separate generated artifacts from source

Create a clear artifact directory such as:

```text
artifacts/
  tokenizer.json
  token_stream.npy
  token_stream.meta.json
  chatbot_transformer.pt
```

Do not place token streams and model checkpoints in the `Embedding` source directory.

### 13.3 Avoid silent deletion

The current startup can delete an SQLite cache automatically. Replace implicit deletion with explicit cache validation and an explicit `--rebuild` operation. Print exactly which artifacts will be replaced.

---

## 14. File-by-File Change Checklist

### `README.md`

- Add environment setup.
- Add preparation, training, terminal, GUI, and test commands.
- Document artifact locations and expected runtime behavior.
- Explain that the first model training can take substantial time.

### `requirements.txt` or `pyproject.toml`

- Declare Python compatibility.
- Declare runtime and test dependencies.
- Remove MySQL dependency after GUI repair.

### `.gitignore`

- Ignore environments, caches, checkpoints, token streams, local databases, and secrets.

### `Pipeline/__init__.py`

- Add package marker.
- Optionally export `Ingest` and `Tokenizer` as the public pipeline API.

### `Pipeline/Metrics.py`

- Add every emitted schema type, including `cornell_movies`.
- Add tests that prevent unknown metric keys.

### `Pipeline/ingest.py`

- Use relative imports.
- Replace broad filename substring routing with an explicit allowlist.
- Parse only `movie_lines.txt` as Cornell lines.
- Add dedicated personality and serialized-dialogue CSV handlers.
- Emit role-formatted, EOS-terminated conversations.
- Keep dataset splits separate.
- Surface required-source failures.

### `Pipeline/tokenizer.py`

- Use relative import for `Ingest`.
- Stop manually inserting `Ġ` before byte encoding.
- Preserve exact decoded whitespace.
- Reject or expose invalid byte symbols instead of silently dropping them.
- Remove every special token when requested.
- Standardize role-token spellings.
- Add tokenizer version/hash metadata.

### `main_and_eval.py`

- Remove unused imports.
- Split preparation, training, and chat into explicit commands.
- Remove SQLite or fix its parameter placeholders and transactions.
- Restore complete batch construction and batch iteration.
- Validate stream length and token bounds.
- Track training and validation loss.
- Save a metadata-rich checkpoint.
- Require compatible tokenizer/checkpoint artifacts for chat.
- Use exact role-token IDs without fallbacks.
- Use the shared `ChatService` for terminal chat.

### `Model/Transformer.py`

- Add configuration and input invariant checks.
- Store model configuration on the instance.
- Keep one PyTorch embedding and one tied output head.
- Verify checkpoint compatibility on load.

### `Model/Generator.py`

- Return generated IDs only.
- Validate top-k, top-p, and temperature.
- Mask invalid control tokens.
- Stop on EOS/role boundaries.
- Add optional repetition controls after correctness tests pass.

### `Model/Attention.py` and `Model/Layers.py`

- Remove from the active application path.
- Move to `legacy/` or delete after confirming nothing depends on them.

### `Embedding/DenseEmbeddingEngine.py` and `Embedding/TrainingSampler.py`

- Remove from the active application path.
- Do not combine them with the PyTorch transformer.
- Move to `legacy/` or delete after migration.

### `chat_backend.py` (new)

- Add `ChatService`.
- Load and validate tokenizer/checkpoint once.
- Own formatting, history, generation, and decoding.
- Expose `reply` and `reset`.

### `chat_gui.py`

- Remove MySQL, dynamic vocabulary, NumPy embeddings, and obsolete model imports.
- Remove the invalid generation expression and obsolete arguments.
- Call `ChatService.reply` in a worker thread.
- Make widget updates through `root.after`.
- Add reset and busy-state behavior.

### `test/Pipeline_test.py`

- Rename to `test/test_pipeline.py`.
- Correct package imports and class names.
- Replace absolute `/Dataset` with fixtures or a repository-relative path.
- Avoid reprocessing the complete dataset in every individual test.

### New test files

Recommended layout:

```text
test/
  test_imports.py
  test_ingest.py
  test_tokenizer.py
  test_cache.py
  test_transformer.py
  test_training.py
  test_generator.py
  test_chat_backend.py
```

---

## 15. Implementation Order

### Phase A: Make the code importable

1. Create the Python environment and dependency manifest.
2. Add package `__init__.py` files.
3. Correct relative imports.
4. Repair test imports and paths.
5. Confirm all non-GUI modules import.

**Exit condition:** Import tests pass.

### Phase B: Make text correct

1. Fix metrics keys.
2. Implement explicit dataset routing.
3. Implement conversation formatting.
4. Fix tokenizer byte handling.
5. Standardize special tokens.
6. Add round-trip and ingestion tests.

**Exit condition:** Clean conversational records are produced, and all tokenizer round-trips pass without `Ġ` artifacts.

### Phase C: Make training work

1. Simplify or repair the cache.
2. Restore batch creation and iteration.
3. Add stream/model invariant checks.
4. Overfit a tiny corpus.
5. Save and reload a checkpoint.
6. Train the full model with validation tracking.

**Exit condition:** Loss decreases, and a compatible checkpoint reloads successfully.

### Phase D: Produce a readable terminal response

1. Correct the generation return value.
2. Match training prompt formatting.
3. Mask unwanted special tokens.
4. Decode generated IDs.
5. Verify a non-empty artifact-free terminal response.

**Exit condition:** The terminal chatbot satisfies the readable-response checks.

### Phase E: Repair the GUI

1. Create the shared `ChatService`.
2. Move terminal chat onto the service.
3. Replace the GUI backend with the same service.
4. Remove MySQL and NumPy model code from the GUI.
5. Verify responsiveness and multi-turn history.

**Exit condition:** Terminal and GUI use the same trained tokenizer/model pipeline.

---

## 16. Final Acceptance Criteria

The minimum remediation is complete only when all of the following are true:

- A documented Python environment can be created from scratch.
- All active modules import without path manipulation errors.
- No source file has syntax errors.
- Required dataset handlers emit non-empty conversational records.
- Metadata and URL files are excluded from dialogue training.
- `decode(encode(text)) == text` for ASCII, whitespace, punctuation, Unicode, and emoji samples.
- Training and inference use identical role tokens and tokenizer assets.
- Every training token ID is within the model vocabulary.
- The training loop constructs real input/target batches.
- Training loss decreases on a tiny-corpus overfit test.
- A checkpoint is saved with configuration and tokenizer metadata.
- A fresh process reloads that checkpoint successfully.
- Terminal generation returns at least one visible non-special token.
- Terminal output contains no raw `Ġ`, `�`, role markers, or repeated unknown-token artifacts.
- The GUI has no database dependency and does not create vocabulary entries dynamically.
- Terminal and GUI both call the same backend service.
- Automated tests pass.
- No credentials remain in source control.

---

## 17. Practical Definition of Success

After the fixes, a normal workflow should look like this:

```text
1. Install dependencies.
2. Prepare the clean dataset and tokenizer.
3. Build the token stream.
4. Train and save the model checkpoint.
5. Start terminal chat and verify readable output.
6. Start the GUI, which loads the same artifacts.
```

The key principle is consistency: one dataset format, one tokenizer, one fixed vocabulary, one PyTorch model, one saved checkpoint, and one inference backend shared by every interface.
