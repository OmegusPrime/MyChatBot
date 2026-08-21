"""Tkinter interface for the trained MyChatBot backend."""

from __future__ import annotations

import argparse
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

from chat_backend import ChatService

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger(__name__)


class ChatBotGUI:
    def __init__(self, root: tk.Tk, artifact_dir: Path, device: str = "auto"):
        self.root = root
        self.root.title("MyChatBot")
        self.root.geometry("520x640")
        self.root.minsize(400, 480)
        self.root.configure(bg="#1e1e1e")
        self.service: ChatService | None = None
        self._busy = False

        self._setup_ui()
        try:
            self.service = ChatService(artifact_dir=artifact_dir, device=device)
        except Exception as error:
            log.exception("chat backend initialization failed")
            self._append_message(
                "System",
                f"The trained chatbot could not be loaded: {error}",
            )
            self._set_busy(True)
        else:
            self._append_message("System", "Model loaded. You can start chatting.")
            self.entry_field.focus_set()

    def _setup_ui(self) -> None:
        self.chat_display = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="white",
            font=("Arial", 11),
            state=tk.DISABLED,
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.tag_config("user", foreground="#4fc1ff", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("bot", foreground="#9cdcfe")
        self.chat_display.tag_config("system", foreground="#6a9955", font=("Arial", 9, "italic"))

        input_frame = tk.Frame(self.root, bg="#1e1e1e")
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.entry_field = tk.Entry(
            input_frame,
            bg="#3d3d3d",
            fg="#ffffff",
            insertbackground="white",
            font=("Arial", 11),
            relief=tk.FLAT,
        )
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.entry_field.bind("<Return>", self._on_send)

        self.send_button = tk.Button(
            input_frame,
            text="Send",
            bg="#007acc",
            fg="#ffffff",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            command=self._on_send,
        )
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0), ipady=5, ipadx=12)

        self.reset_button = tk.Button(
            input_frame,
            text="Reset",
            bg="#555555",
            fg="#ffffff",
            relief=tk.FLAT,
            command=self._on_reset,
        )
        self.reset_button.pack(side=tk.RIGHT, padx=(5, 0), ipady=5, ipadx=8)

    def _append_message(self, sender: str, text: str) -> None:
        self.chat_display.config(state=tk.NORMAL)
        if sender == "You":
            self.chat_display.insert(tk.END, f"\nYou: {text}\n", "user")
        elif sender == "Chatbot":
            self.chat_display.insert(tk.END, f"\nChatbot: {text}\n", "bot")
        else:
            self.chat_display.insert(tk.END, f"[{text}]\n", "system")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.send_button.config(state=state)
        self.reset_button.config(state=state)
        self.entry_field.config(state=state)
        if not busy:
            self.entry_field.focus_set()

    def _on_send(self, event=None) -> str | None:
        if self._busy or self.service is None:
            return "break" if event is not None else None
        user_text = self.entry_field.get().strip()
        if not user_text:
            return "break" if event is not None else None

        self.entry_field.delete(0, tk.END)
        self._append_message("You", user_text)
        self._set_busy(True)
        threading.Thread(
            target=self._generate_response,
            args=(user_text,),
            daemon=True,
        ).start()
        return "break" if event is not None else None

    def _generate_response(self, user_text: str) -> None:
        try:
            assert self.service is not None
            response = self.service.reply(user_text)
        except Exception as error:
            log.exception("response generation failed")
            self.root.after(0, self._finish_response, "System", f"Response failed: {error}")
        else:
            self.root.after(0, self._finish_response, "Chatbot", response)

    def _finish_response(self, sender: str, text: str) -> None:
        self._append_message(sender, text)
        self._set_busy(False)

    def _on_reset(self) -> None:
        if self._busy or self.service is None:
            return
        self.service.reset()
        self._append_message("System", "Conversation reset.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start the MyChatBot desktop interface")
    parser.add_argument("--artifact-dir", type=Path, default=BASE_DIR / "artifacts")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    ChatBotGUI(root, artifact_dir=args.artifact_dir, device=args.device)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
