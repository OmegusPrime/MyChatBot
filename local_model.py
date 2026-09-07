"""Private, automatically managed llama.cpp CPU server; standard library only.

The pinned setup script installs the runtime and model. This module never
downloads anything and never sends a conversation outside the loopback interface.
"""
from __future__ import annotations

import atexit
from collections import deque
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time


class _ServerHTTPError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class LocalModelProvider:
    """One owned server process, authenticated random port, bounded HTTP requests."""

    def __init__(
        self, project_dir: str | Path, *, context_length: int = 4096,
        startup_timeout: float = 120, request_timeout: float = 180,
    ):
        if not isinstance(context_length, int) or not 512 <= context_length <= 32768:
            raise ValueError("context_length must be between 512 and 32768")
        if startup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("Model timeouts must be positive")
        self.project_dir = Path(project_dir).resolve()
        self.context_length = context_length
        self.request_timeout = request_timeout
        self._closed = threading.Event()
        self._process_lock = threading.Lock()
        self._process = None
        self._stderr_thread = None
        self._startup_lines: deque[str] = deque(maxlen=12)
        self._api_key = secrets.token_urlsafe(32)
        self._exact_token_count = True
        executable = self.project_dir / "runtime" / "llama" / "llama-server.exe"
        model = self.project_dir / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        missing = [str(path) for path in (executable, model) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Local chatbot setup is incomplete. Run setup_chatbot.ps1 from "
                f"{self.project_dir}, then start the chatbot again. Missing: "
                + ", ".join(missing)
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        # Do not inherit llama.cpp options that could enable networking or tools.
        environment = {
            key: value for key, value in os.environ.items()
            if not key.upper().startswith("LLAMA_")
        }
        environment["LLAMA_API_KEY"] = self._api_key
        args = [
            str(executable), "--model", str(model), "--alias", "mychatbot",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--ctx-size", str(context_length), "--parallel", "1",
            "--n-gpu-layers", "0", "--threads", str(min(8, max(1, os.cpu_count() or 1))),
            "--batch-size", "512", "--ubatch-size", "128",
            "--no-webui", "--no-slots", "--no-context-shift", "--jinja",
        ]
        try:
            self._process = subprocess.Popen(
                args, cwd=str(self.project_dir), env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, name="local-model-diagnostics", daemon=True
            )
            self._stderr_thread.start()
            deadline = time.monotonic() + startup_timeout
            last_error = None
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    detail = "\n".join(self._startup_lines)
                    raise RuntimeError(
                        f"The local model could not start (exit {self._process.returncode}). "
                        "Run setup_chatbot.ps1 to repair the runtime and model."
                        + (f"\n{detail}" if detail else "")
                    )
                try:
                    health = self._request("GET", "/health", timeout=1)
                    if health.get("status") == "ok":
                        # /health can be unauthenticated. Checking /props proves
                        # this is the process with our per-launch secret.
                        props = self._request("GET", "/props", timeout=2)
                        actual_context = props.get("default_generation_settings", {}).get("n_ctx")
                        if isinstance(actual_context, int) and actual_context > 0:
                            self.context_length = min(context_length, actual_context)
                        break
                except RuntimeError as error:
                    last_error = error
                self._closed.wait(0.2)
            else:
                raise RuntimeError(
                    "The local model took too long to load. Close other memory-heavy "
                    "applications and try again."
                ) from last_error
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                # Diagnostics are bounded and held only in RAM, never chat logs.
                self._startup_lines.append(line.strip()[:600])
        except (OSError, ValueError):
            pass

    def _request(
        self, method: str, path: str, payload: dict | None = None,
        *, timeout: float | None = None,
    ) -> dict:
        if self._closed.is_set():
            raise RuntimeError("The local model is closed.")
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("The local model stopped. Restart the chatbot.")
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=timeout or self.request_timeout
        )
        try:
            body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            connection.request(
                method, path, body=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise RuntimeError("The local model returned an unexpectedly large response.")
            if not 200 <= response.status < 300:
                # Do not follow redirects or forward any request to another host.
                raise _ServerHTTPError(
                    response.status,
                    f"The local model returned HTTP {response.status}. "
                    "Try a shorter message or restart the chatbot.",
                )
            try:
                decoded = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as error:
                raise RuntimeError("The local model returned an invalid response. Restart the chatbot.") from error
            if not isinstance(decoded, dict):
                raise RuntimeError("The local model returned an invalid response object.")
            if decoded.get("error"):
                raise RuntimeError("The local model reported an error. Try again or restart the chatbot.")
            return decoded
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                "The local model timed out. Your message was not added to the conversation. "
                "Try a shorter message or restart the chatbot."
            ) from error
        except (http.client.HTTPException, OSError) as error:
            raise RuntimeError("Cannot reach the local model. Restart the chatbot and try again.") from error
        finally:
            connection.close()

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count the model's actual chat template, including assistant prefix/BOS."""
        if self._exact_token_count:
            try:
                formatted = self._request("POST", "/apply-template", {"messages": messages})
                prompt = formatted.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise RuntimeError("The model returned an invalid chat template.")
                result = self._request(
                    "POST", "/tokenize",
                    {"content": prompt, "add_special": True, "parse_special": True},
                )
                tokens = result.get("tokens")
                if not isinstance(tokens, list) or not tokens:
                    raise RuntimeError("The model returned an invalid token count.")
                return len(tokens)
            except _ServerHTTPError as error:
                if error.status not in {404, 501}:
                    raise
                self._exact_token_count = False
        # Older llama.cpp builds may lack /apply-template. Qwen's byte-level
        # tokenizer cannot use more content tokens than UTF-8 bytes. The fixed
        # overhead comfortably bounds ChatML separators and the generation prefix.
        return 256 + sum(len(message["content"].encode("utf-8")) + 64 for message in messages)

    def complete(
        self, messages: list[dict[str, str]], *, max_new_tokens: int, temperature: float
    ) -> str:
        result = self._request(
            "POST", "/v1/chat/completions",
            {
                "model": "mychatbot", "messages": messages, "stream": False,
                "max_tokens": max_new_tokens, "temperature": temperature,
                "top_p": 0.9, "repeat_penalty": 1.05,
                "cache_prompt": True,
            },
        )
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("The local model did not return a reply. Please try again.")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The local model returned an empty reply. Please try again.")
        if choices[0].get("finish_reason") == "length":
            return content.strip() + "\n\n[Reply reached the length limit. Ask me to continue if needed.]"
        return content.strip()

    def close(self) -> None:
        """Terminate only the server process that this provider started."""
        with self._process_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            atexit.unregister(self.close)
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    # The server can exit between poll() and terminate().
                    if process.poll() is None:
                        process.kill()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=0.5)
            if process is not None and process.stderr is not None:
                process.stderr.close()


__all__ = ["LocalModelProvider"]
