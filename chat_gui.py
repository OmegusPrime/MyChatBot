"""Responsive desktop chat using the same conversational service as the CLI."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

from chat_backend import ChatService

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger(__name__)


class ChatBotGUI:
    def __init__(
        self, root: tk.Tk, artifact_dir: Path, device: str = "auto", backend: str = "local"
    ):
        self.root = root
        self.artifact_dir = artifact_dir
        self.device = device
        self.backend = backend
        self.root.title("MyChatBot")
        self.root.geometry("720x760")
        self.root.minsize(460, 500)
        self.root.configure(bg="#111827")
        self.service: ChatService | None = None
        self._busy = True
        self._events: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._service_lock = threading.Lock()
        self._poll_id = None

        self._setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_id = self.root.after(50, self._drain_events)
        self._start_loading()

    def _setup_ui(self) -> None:
        header = tk.Frame(self.root, bg="#111827")
        header.pack(fill=tk.X, padx=20, pady=(18, 8))
        tk.Label(
            header, text="MyChatBot", fg="#f9fafb", bg="#111827",
            font=("Segoe UI", 20, "bold"),
        ).pack(side=tk.LEFT)
        self.reset_button = tk.Button(
            header, text="New chat", command=self._on_reset,
            bg="#374151", fg="#f9fafb", relief=tk.FLAT, padx=12, pady=6,
        )
        self.reset_button.pack(side=tk.RIGHT)

        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, bg="#1f2937", fg="#f9fafb",
            insertbackground="white", font=("Segoe UI", 11),
            relief=tk.FLAT, borderwidth=0, padx=16, pady=12, state=tk.DISABLED,
        )
        self.chat_display.pack(padx=20, pady=8, fill=tk.BOTH, expand=True)
        self.chat_display.tag_config("user", foreground="#93c5fd", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("bot", foreground="#f9fafb")
        self.chat_display.tag_config("system", foreground="#a7b5c9", font=("Segoe UI", 10))

        status_frame = tk.Frame(self.root, bg="#111827")
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 8))
        self.status_label = tk.Label(
            status_frame, text="Starting...", anchor=tk.W, bg="#111827", fg="#a7b5c9",
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.retry_button = tk.Button(
            status_frame, text="Retry loading", command=self._start_loading,
            bg="#374151", fg="#f9fafb", relief=tk.FLAT,
        )

        input_frame = tk.Frame(self.root, bg="#111827")
        input_frame.pack(padx=20, pady=(0, 20), fill=tk.X)
        self.entry_field = tk.Entry(
            input_frame, bg="#374151", fg="#ffffff", insertbackground="white",
            font=("Segoe UI", 11), relief=tk.FLAT,
        )
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)
        self.entry_field.bind("<Return>", self._on_send)
        self.send_button = tk.Button(
            input_frame, text="Send", bg="#2563eb", fg="#ffffff",
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT,
            command=self._on_send, padx=18, pady=8,
        )
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0))

    def _append_message(self, sender: str, text: str) -> None:
        self.chat_display.config(state=tk.NORMAL)
        tag = "user" if sender == "You" else "bot" if sender == "Chatbot" else "system"
        prefix = f"{sender}: " if sender != "System" else ""
        self.chat_display.insert(tk.END, f"{prefix}{text}\n\n", tag)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        state = tk.DISABLED if busy or self.service is None else tk.NORMAL
        self.send_button.config(state=state)
        self.reset_button.config(state=state)
        self.entry_field.config(state=state)
        self.status_label.config(text=status)
        if state == tk.NORMAL:
            self.entry_field.focus_set()

    def _start_loading(self) -> None:
        if self._closed.is_set():
            return
        self.retry_button.pack_forget()
        self._set_busy(True, "Loading the local model. The first start can take a minute...")
        # A non-daemon worker cleans up a late-created server after window close.
        threading.Thread(target=self._load_service, name="chat-load", daemon=False).start()

    def _load_service(self) -> None:
        service = None
        try:
            service = ChatService(
                artifact_dir=self.artifact_dir, device=self.device, backend=self.backend
            )
            with self._service_lock:
                if not self._closed.is_set():
                    self.service = service
                    self._events.put(("ready", service.status))
                    return
            service.close()
        except Exception as error:
            log.exception("chat backend initialization failed")
            if service is not None:
                self._close_service(service)
            self._events.put(("load_error", str(error)))

    def _drain_events(self) -> None:
        """Only this main-thread timer updates widgets after worker operations."""
        if self._closed.is_set():
            return
        while True:
            try:
                event, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if event == "ready":
                self._append_message("System", "Ready. Say hello or ask a question.")
                self._set_busy(False, payload)
            elif event == "load_error":
                self._append_message("System", f"Could not start the chatbot. {payload}")
                self._set_busy(False, "Startup failed. Fix the issue above, then retry.")
                self.retry_button.pack(side=tk.RIGHT, padx=(8, 0))
            elif event == "reply":
                self._append_message("Chatbot", payload)
                self._set_busy(False, self.service.status if self.service else "Unavailable")
            elif event == "reply_error":
                self._append_message("System", f"Could not answer. {payload}")
                self._set_busy(False, "Ready to try again")
            elif event == "reset":
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.delete("1.0", tk.END)
                self.chat_display.config(state=tk.DISABLED)
                self.entry_field.config(state=tk.NORMAL)
                self.entry_field.delete(0, tk.END)
                self._append_message("System", "New conversation. What would you like to talk about?")
                self._set_busy(False, self.service.status if self.service else "Unavailable")
        self._poll_id = self.root.after(50, self._drain_events)

    def _on_send(self, event=None) -> str | None:
        result = "break" if event is not None else None
        if self._busy or self.service is None or self._closed.is_set():
            return result
        user_text = self.entry_field.get().strip()
        if not user_text:
            return result
        self.entry_field.delete(0, tk.END)
        self._append_message("You", user_text)
        self._set_busy(True, "Thinking...")
        threading.Thread(
            target=self._generate_response, args=(self.service, user_text),
            name="chat-reply", daemon=False,
        ).start()
        return result

    def _generate_response(self, service: ChatService, user_text: str) -> None:
        try:
            self._events.put(("reply", service.reply(user_text)))
        except Exception as error:
            if not self._closed.is_set():
                log.exception("response generation failed")
                self._events.put(("reply_error", str(error)))

    def _on_reset(self) -> None:
        if self._busy or self.service is None or self._closed.is_set():
            return
        self._set_busy(True, "Starting a new conversation...")
        threading.Thread(
            target=self._reset_service, args=(self.service,), name="chat-reset", daemon=False,
        ).start()

    def _reset_service(self, service: ChatService) -> None:
        try:
            service.reset()
            self._events.put(("reset", None))
        except Exception as error:
            self._events.put(("reply_error", f"Could not reset the conversation: {error}"))

    @staticmethod
    def _close_service(service: ChatService) -> None:
        try:
            service.close()
        except Exception:
            log.exception("chat backend cleanup failed")

    def _on_close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
        with self._service_lock:
            service, self.service = self.service, None
        if service is not None:
            threading.Thread(
                target=self._close_service, args=(service,), name="chat-close", daemon=False,
            ).start()
        self.root.destroy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start the MyChatBot desktop interface")
    parser.add_argument("--artifact-dir", type=Path, default=BASE_DIR / "artifacts")
    parser.add_argument("--device", default="auto", help="device for the legacy training model")
    parser.add_argument("--backend", choices=("local", "legacy"), default="local")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    app = ChatBotGUI(root, artifact_dir=args.artifact_dir, device=args.device, backend=args.backend)
    try:
        root.mainloop()
    finally:
        app._on_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
