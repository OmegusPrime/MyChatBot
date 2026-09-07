# MyChatBot repair record

Date: 2026-09-07. Source project: `D:\ChatBot\MyChatBot`.

## Objective and diagnosis

Make the application start reliably and answer ordinary conversational requests, including follow-up questions. The old default backend could not provide this: it required absent training artifacts, handled only a few fixed intents, retrieved loosely matching movie dialogue, and returned a clarification sentence for everything else. Its tiny transformer and 32-token conversation window were insufficient for general chat.

The earlier environment diagnosis was rechecked. A newer Python 3.14.7 virtual environment now exists in the source repository and can load the installed training dependencies and Tkinter. It was not replaced. The configured task directory `D:\MyChatBot` remains stale; all installed repairs target the actual source project above. Existing IDE edits are preserved.

## Changes

| Area | Repair | Result |
| --- | --- | --- |
| Default response engine | Use an instruction-tuned Qwen model through local llama.cpp | Generate a response to the actual question |
| Startup dependencies | Remove eager PyTorch/checkpoint imports from normal chat | Chat no longer needs the research training artifacts |
| Conversation formatting | Use model chat template with system/user/assistant roles | The model receives the right speaker boundaries |
| Memory | Retain complete exchanges and fit the model context budget | Follow-up questions can refer to recent messages |
| Context overflow | Trim oldest complete exchanges; reject an individually oversized message | No silent truncation of the current request |
| Failed replies | Update history only after successful generation | Retry does not inherit a failed or empty assistant answer |
| Retrieval and fixed intents | Remove dialogue lookup from the default response path | No unrelated movie quote caused by keyword overlap |
| Runtime process | Start a hidden, authenticated loopback server for the session | No API key, fixed port, or manually managed service |
| Cleanup | Close the server when the session ends | Avoid an orphan model process consuming memory |
| Desktop threading | Load/generate on workers; update widgets through the UI event queue | The window stays responsive and avoids cross-thread Tk calls |
| Desktop recovery | Loading/thinking status and Retry after startup failure | Clear feedback and recovery without editing code |
| Terminal | Standalone standard-library chat entry point | Both interfaces share the same conversation service |
| Launching | Project-relative PowerShell launcher and double-click entry point | Start from Explorer or another working directory |
| Installation | Pinned portable runtime and checksum-verified resumable model download | Reproducible setup after cloning or moving |
| Original model | Preserve it as explicit `--backend legacy` | Research training remains usable independently |
| Documentation | Rewrite current README and retain the historical fix plan | Startup instructions describe the implemented application |

## Installed assets and provenance

- Model: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes.
- Model SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
- Runtime: official `llama-b10826-bin-win-cpu-x64.zip` (18,412,429 bytes), extracted under `runtime/llama/`.
- Runtime archive SHA-256: `5828cccc7261b14607d23de3144f35fac4249d9fd207e13bff5e31dd8ae39d56`.
- Upstream sources: [Qwen GGUF repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) and [llama.cpp b10826](https://github.com/ggml-org/llama.cpp/releases/tag/b10826).

The setup script embeds these hashes, preserves an unexpected existing file, and verifies downloads before using them. Binaries, weights, and transient runtime data are ignored by Git. They can be copied during a move or reproduced with setup.

## Validation

The complete suite ran from the installed project using its own virtual environment:

```text
52 passed, 51 subtests passed in 20.22s
```

This includes the original tokenizer, ingestion, preparation, retrieval, transformer, and tiny training/save/load checks; the new conversation and transport regressions; real Tk widget tests; and terminal entry-point tests. Desktop tests cover retry after missing-model errors and cleanup when the window closes during loading or generation. HTTP tests cover empty/malformed replies, redirects, errors, timeouts, context counting, and process shutdown.

The installed model was also exercised directly, through the real desktop event loop, and through `main_and_eval.py --chat`. These are observed smoke-test results, not a broad model-quality benchmark:

| Input or action | Observed result |
| --- | --- |
| `Hi` | `Hello! How can I assist you today?` |
| `Hi! What is 2 + 2?` | `2 + 2 equals 4.` |
| Introduce the name Dhruv, then ask `What is my name?` | `Your name is Dhruv.` |
| Ask why the sky is blue in two sentences | Explained shorter-wavelength scattering and named Rayleigh scattering |
| Ask for a Python function to reverse a string | Returned a function using `s[::-1]` |
| Reset, then ask for the user's name | Explained that the name was not known; the prior name did not leak through reset |
| Desktop: ask for the capital of France | Displayed `The capital of France is Paris.` |
| Desktop: New chat | Cleared displayed conversation and backend history |
| Desktop/session close | Owned model process exited |
| Existing terminal command: Hi, RESET, arithmetic, END | Answered, reset, answered correctly, and exited successfully |

Initial loading took approximately 5.75 seconds in the direct test. Its simple replies took about 0.59–4.21 seconds on this machine's CPU; timings depend on prompt length, model cache, and other applications. The model file and runtime archive passed their pinned SHA-256 checks before use.

The automated tests run without downloading a model. The real-model checks above were performed separately against the installed weights. No training-artifact reconstruction or costly research-model retraining was necessary for normal chat.

## Changed files

- `chat_backend.py`: shared local conversation service and explicit legacy selection.
- `local_model.py`: private server startup, HTTP communication, template token counting, and process cleanup.
- `legacy_backend.py`: preserved original trained backend with cleanup support.
- `chat_gui.py`: responsive desktop loading, send/reset, errors, retry, and closing behavior.
- `chat_cli.py`: standalone terminal loop with reset, exit, EOF, and interrupt handling.
- `main_and_eval.py`: shared chat command and deferred training imports.
- `setup_chatbot.ps1`: resumable verified local runtime/model installation and check command.
- `launch_chatbot.ps1` and `Start Chatbot.cmd`: project-relative startup and interpreter validation.
- `.gitignore`: exclude downloaded runtime, model weights, and transient files.
- `test/test_chat_service.py`, `test/test_local_model.py`, `test/test_chat_gui.py`, and `test/test_chat_cli.py`: new regression coverage.
- `test/test_model.py` and `test/test_training_integration.py`: exercise the preserved training model explicitly and release its resources.
- `README.md` and this document: current instructions, verified results, and limitations.

## Limits that remain

This repair supplies a functioning local assistant; it does not make a 1.5-billion-parameter model equivalent to a frontier hosted model. Answers can be wrong, especially for complex reasoning and specialist facts. The application has no live browsing, document index, tool execution, or memory across sessions. Long chats lose their oldest exchanges as the context fills. The default answer length is bounded to keep CPU latency reasonable.

Future capability work can add a larger model suited to the available hardware, curated document retrieval with sources, and task-specific evaluation. Those capabilities require separate implementation and are not claimed as completed by this repair.
