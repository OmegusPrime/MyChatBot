"""Conversational service shared by the desktop and terminal clients."""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Protocol

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"
SYSTEM_PROMPT = (
    "You are MyChatBot, a helpful conversational assistant running locally on the user's computer. "
    "Respond directly to the user's latest request, using relevant details from this conversation. "
    "Be friendly, clear and concise. Follow requested formats and explain things simply. "
    "Ask a short clarifying question when needed. If you do not know something, say so; "
    "do not invent facts or pretend to have searched the internet, accessed files or used tools. "
    "You have no live internet access or tools."
)


class ChatProvider(Protocol):
    context_length: int

    def count_tokens(self, messages: list[dict[str, str]]) -> int: ...

    def complete(
        self, messages: list[dict[str, str]], *, max_new_tokens: int, temperature: float
    ) -> str: ...

    def close(self) -> None: ...


class ChatService:
    """Remember successful complete exchanges; import the training experiment only on demand."""

    def __init__(
        self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        device: str = "auto", backend: str = "local",
        project_dir: str | Path | None = None, *, provider: ChatProvider | None = None,
    ):
        if backend not in {"local", "legacy"}:
            raise ValueError("backend must be 'local' or 'legacy'")
        if backend == "legacy" and provider is not None:
            raise ValueError("A provider can only be used with the local backend")
        self.backend = backend
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = threading.Event()
        self._history: list[dict[str, str]] = []
        self._legacy = None
        self._provider: ChatProvider | None = None
        if backend == "legacy":
            try:
                from legacy_backend import LegacyChatService
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "The experimental backend requires the training dependencies. "
                    "Install requirements.txt, or use the default local chatbot."
                ) from error
            self._legacy = LegacyChatService(artifact_dir=artifact_dir, device=device)
            self.context_length = self._legacy.context_length
        else:
            if device not in {"auto", "cpu"}:
                raise ValueError("The bundled local runtime uses CPU; use device='auto' or 'cpu'")
            if provider is None:
                from local_model import LocalModelProvider
                provider = LocalModelProvider(project_dir=project_dir or BASE_DIR)
            self._provider = provider
            self.context_length = provider.context_length
            if not isinstance(self.context_length, int) or self.context_length < 256:
                provider.close()
                raise ValueError("The chat provider must support at least 256 context tokens")

    @property
    def status(self) -> str:
        if self._closed.is_set():
            return "Chatbot closed"
        if self._legacy is not None:
            return self._legacy.status
        return "Ready — local Qwen2.5 1.5B Instruct · CPU · chats stay on this computer"

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a snapshot so callers cannot corrupt conversation roles."""
        with self._lock:
            return [dict(message) for message in self._history]

    def _require_open(self) -> None:
        if self._closed.is_set():
            raise RuntimeError("The chatbot is closed. Start it again to chat.")

    def reset(self) -> None:
        with self._lock:
            self._require_open()
            self._history.clear()
            if self._legacy is not None:
                self._legacy.reset()

    def close(self) -> None:
        """Stop our server even if another thread is waiting for a response."""
        with self._state_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            if self._provider is not None:
                self._provider.close()
            if self._legacy is not None:
                self._legacy.close()
        with self._lock:
            self._history.clear()

    def reply(
        self, user_text: str, *, max_new_tokens: int = 384,
        temperature: float = 0.3, **legacy_options,
    ) -> str:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("Message cannot be empty and must be text")
        cleaned = user_text.strip()
        with self._lock:
            self._require_open()
            if self._legacy is not None:
                return self._legacy.reply(
                    cleaned, max_new_tokens=max_new_tokens,
                    temperature=temperature, **legacy_options,
                )
            unknown = set(legacy_options) - {"top_k", "top_p", "allow_generative_fallback"}
            if unknown:
                raise TypeError("Unknown reply options: " + ", ".join(sorted(unknown)))
            maximum = min(2048, self.context_length - 128)
            if (
                isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= maximum
            ):
                raise ValueError(f"max_new_tokens must be an integer between 1 and {maximum}")
            if (
                isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or not 0 <= temperature <= 2
            ):
                raise ValueError("temperature must be a finite number between 0 and 2")
            if len(cleaned.encode("utf-8")) > self.context_length * 8:
                raise ValueError("Your message is too long. Please shorten it or split it into parts.")
            assert self._provider is not None
            system = {"role": "system", "content": SYSTEM_PROMPT}
            user = {"role": "user", "content": cleaned}
            retained = [dict(message) for message in self._history[-46:]]
            budget = self.context_length - max_new_tokens - 16
            messages = [system, *retained, user]
            while self._provider.count_tokens(messages) > budget:
                if not retained:
                    raise ValueError(
                        "Your message is too long for the current response limit. "
                        "Please shorten it, split it into parts, or reduce max_new_tokens."
                    )
                del retained[:2]
                messages = [system, *retained, user]
            response = self._provider.complete(
                [dict(message) for message in messages],
                max_new_tokens=max_new_tokens, temperature=float(temperature),
            )
            if not isinstance(response, str) or not response.strip():
                raise RuntimeError("The local model returned an empty reply. Please try again.")
            self._require_open()
            response = response.strip()
            # Commit only after success. Failed requests never remove context or
            # leave a partial user turn behind.
            self._history = [*retained, user, {"role": "assistant", "content": response}]
            return response

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


__all__ = ["ChatService", "ChatProvider", "DEFAULT_ARTIFACT_DIR", "SYSTEM_PROMPT"]
