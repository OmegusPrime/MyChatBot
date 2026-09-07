"""Terminal chat. Local mode needs only the standard library and model assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from chat_backend import ChatService

BASE_DIR = Path(__file__).resolve().parent


def run_terminal_chat(
    artifact_dir: str | Path = BASE_DIR / "artifacts",
    device: str = "auto",
    backend: str = "local",
) -> int:
    service = None
    try:
        print("Starting MyChatBot. Loading the model can take a minute...", flush=True)
        service = ChatService(artifact_dir=artifact_dir, device=device, backend=backend)
        print(f"{service.status}\n")
        print("Type /exit or END to quit; /reset or RESET starts a new conversation.\n")
        while True:
            user_text = input("You: ").strip()
            if user_text.lower() in {"end", "/exit", "/quit"}:
                return 0
            if user_text.lower() in {"reset", "/reset"}:
                service.reset()
                print("Conversation reset.\n")
                continue
            if not user_text:
                continue
            try:
                print(f"Chatbot: {service.reply(user_text)}\n", flush=True)
            except Exception as error:
                print(f"Could not answer: {error}\n", flush=True)
    except (EOFError, KeyboardInterrupt):
        print("\nChat closed.")
        return 0
    except Exception as error:
        print(f"Could not start or continue MyChatBot: {error}")
        return 1
    finally:
        if service is not None:
            service.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chat with MyChatBot in your terminal")
    parser.add_argument("--artifact-dir", type=Path, default=BASE_DIR / "artifacts")
    parser.add_argument("--device", default="auto", help="device for the legacy training model")
    parser.add_argument("--backend", choices=("local", "legacy"), default="local")
    args = parser.parse_args(argv)
    return run_terminal_chat(args.artifact_dir, args.device, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
