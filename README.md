# MyChatBot

A local desktop and terminal chatbot. The default conversation engine is Qwen2.5-1.5B-Instruct (Q4_K_M), running through a portable llama.cpp server on this computer. It answers questions using the current conversation instead of selecting unrelated movie dialogue. No API key or model training is required for normal chat.

## Start chatting

On this computer, open `D:\ChatBot\MyChatBot` and double-click **Start Chatbot.cmd**. The first model load can take a few seconds. The window shows its loading status, then enables the message box.

- Enter sends a message.
- New chat starts a fresh conversation.
- Retry reloads the backend after a startup failure.
- Closing the window stops its model server.

For terminal chat:

```powershell
cd D:\ChatBot\MyChatBot
.\.venv\Scripts\python.exe chat_cli.py
```

Use `RESET` or `/reset` for a new conversation, and `END` or `/exit` to leave. The existing `main_and_eval.py --chat` command also uses the new backend.

## Setup on another computer or after moving the project

Requirements: x64 Windows, Python 3.11 or newer with Tkinter, approximately 2 GB free disk space, and sufficient free memory for the model. Normal chat uses only the Python standard library; PyTorch is needed only for the training experiment. The current Python 3.14 environment was checked for startup compatibility.

1. Use a working Python installation. If the copied `.venv` references a missing interpreter, create a new environment with that installation; virtual environments are not portable.
2. From the project folder, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_chatbot.ps1
```

Setup downloads about 1.14 GB from the official Qwen and llama.cpp repositories, verifies pinned SHA-256 hashes, and installs into `models/` and `runtime/`. Interrupted downloads can resume. Unexpected existing files are preserved and reported. No Windows service or global model manager is installed.

3. Run **Start Chatbot.cmd**, or `python chat_gui.py`.

To check the downloaded model and runtime later:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_chatbot.ps1 -CheckOnly
```

Paths resolve relative to the source files, so starting from a different working directory is supported. When relocating, copy `models/` and `runtime/` too, or rerun setup; they are intentionally excluded from Git.

## How replies work

`chat_backend.ChatService` sends structured system, user, and assistant messages to the local model. Completed exchanges form the conversation history. The model's own chat template and tokenizer determine the context budget; oldest complete exchanges are removed when necessary. An overlong individual message is rejected with a clear explanation. Failed requests do not become part of the conversation.

The desktop starts and generates replies on workers while all Tkinter operations stay on the main thread. A local server is started for the session on an available loopback port with a random access token. It is stopped on close. The runtime does not need internet access after setup, and this application does not send chat messages to a hosted API or save transcripts to disk.

The small local model can handle everyday conversation, basic explanations, writing, and simple code. It can still make factual or reasoning mistakes. It has no live web access or access to arbitrary files, and remembers only the recent context of the current session. Reset or restart clears it. This is not a claim of large hosted-model quality.

## Training experiment

The original PyTorch transformer, tokenizer, datasets, preparation, and training pipeline remain available as an explicit experimental backend. They are not required to start normal chat.

```powershell
python -m pip install -r requirements.txt
python main_and_eval.py --prepare --train
python chat_cli.py --backend legacy
python chat_gui.py --backend legacy
```

The legacy backend needs a matching `artifacts/tokenizer.json` and `artifacts/chatbot_transformer.pt`; the optional dialogue index can be created with `python main_and_eval.py --index`. Use `--rebuild` only when intentionally rebuilding training artifacts. The legacy model remains too small for reliable general conversation.

## Tests

With training dependencies installed:

```powershell
python -m pytest -q
```

The regression suite checks conversation continuity, reset, context limits, failed-request recovery, HTTP validation, process cleanup, tokenizer round trips, ingestion, retrieval, and the original tiny training/save/load workflow. See [CHATBOT_FIXES.md](CHATBOT_FIXES.md) for the repair record and real-model validation.

## Troubleshooting

- **Missing model/runtime:** run `setup_chatbot.ps1`, then Retry in the window.
- **Copied virtual environment will not start:** recreate it using an installed Python; the launcher tests the project environment and then available Python commands.
- **Slow first response:** allow model loading to finish. Other applications using much of the RAM/CPU may increase latency.
- **Legacy checkpoint error:** omit `--backend legacy` for normal chat, or prepare/train a matching experimental checkpoint.
- **Port conflicts:** each session selects an available local port; no fixed port or separately started server is required.

## Model and runtime sources

- [Official Qwen model and model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [Pinned llama.cpp Windows runtime release](https://github.com/ggml-org/llama.cpp/releases/tag/b10826)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

The historical [PROJECT_FIX_PLAN.md](PROJECT_FIX_PLAN.md) documents the earlier training/tokenizer repairs. The current chat design and setup above supersede its default-response architecture.
