"""Shared trained chatbot backend used by terminal chat and the Tkinter GUI."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import torch

from Model.Generator import ChatBotInferenceEngine
from Model.Transformer import TransformerCoreStack
from Pipeline.tokenizer import SPECIAL_TOKENS, Tokenizer
from response_index import ResponseRetriever, normalize_prompt

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"
SAFE_FALLBACK = "I'm not confident I understood that. Could you rephrase it with a little more detail?"


def intent_response(text: str) -> str | None:
    """Return deterministic responses for clear, common conversational intents."""
    normalized = normalize_prompt(text)
    tokens = normalized.split()
    token_set = set(tokens)
    if not tokens:
        return None

    greeting_words = {
        "hi", "hello", "hey", "hiya", "greetings", "morning", "afternoon", "evening",
        "good", "there", "bot", "chatbot",
    }
    if len(tokens) <= 3 and token_set <= greeting_words and token_set & {
        "hi", "hello", "hey", "hiya", "greetings", "morning", "afternoon", "evening",
    }:
        return "Hi! How can I help you today?"
    if normalized in {"how are you", "how are you doing", "how is it going", "how's it going"}:
        return "I'm doing well, thanks! How can I help you?"
    if token_set & {"thanks", "thank", "thx"} and len(tokens) <= 6:
        return "You're welcome!"
    if token_set & {"bye", "goodbye", "farewell"} and len(tokens) <= 5:
        return "Goodbye! It was nice talking with you."
    if normalized in {
        "who are you",
        "what are you",
        "what is your name",
        "what's your name",
        "tell me your name",
    }:
        return "I'm MyChatBot, a conversational assistant trained on the dialogue in this project."
    if normalized in {"help", "can you help", "can you help me", "i need help"}:
        return "Of course. Tell me what you need help with, and include any useful details."
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class ChatService:
    """Own a fixed tokenizer/model pair and bounded multi-turn history."""

    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR, device: str = "auto"):
        root = Path(artifact_dir).resolve()
        tokenizer_path = root / "tokenizer.json"
        checkpoint_path = root / "chatbot_transformer.pt"
        if not tokenizer_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                "Tokenizer or model checkpoint is missing. Run "
                "'python main_and_eval.py --prepare --train' first."
            )

        self.device = _resolve_device(device)
        self.tokenizer = Tokenizer.from_pretrained(str(tokenizer_path))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise RuntimeError("Checkpoint uses an unsupported format; retrain the model")
        if checkpoint.get("format_version") != 1:
            raise RuntimeError("Checkpoint format is unsupported; retrain the model")
        if checkpoint.get("tokenizer_sha256") != _sha256_file(tokenizer_path):
            raise RuntimeError("Checkpoint and tokenizer do not match; prepare and train again")
        if checkpoint.get("vocab_size") != self.tokenizer.vocab_size:
            raise RuntimeError("Checkpoint vocabulary size does not match the tokenizer")

        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise RuntimeError("Checkpoint does not contain model configuration")
        self.model = TransformerCoreStack(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()

        self.context_length = int(checkpoint.get("context_length", 32))
        self.engine = ChatBotInferenceEngine(self.model, context_length=self.context_length)
        self.user_id = self._required_id("<user>")
        self.assistant_id = self._required_id("<assistant>")
        self.eos_id = self._required_id("<eos>")
        self.pad_id = self._required_id("<pad>")
        self.unk_id = self._required_id("<unk>")
        self.bos_id = self._required_id("<bos>")
        self.sep_id = self._required_id("<sep>")
        self.mask_id = self._required_id("<mask>")
        self.whitespace_only_ids = self._whitespace_only_ids()
        response_index_path = root / "response_pairs.db"
        self.retriever = ResponseRetriever(response_index_path) if response_index_path.exists() else None
        self.history: list[int] = []
        self._lock = threading.Lock()

    def _required_id(self, token: str) -> int:
        try:
            return int(self.tokenizer.vocab[token])
        except KeyError as error:
            raise RuntimeError(f"Tokenizer is missing required token {token}") from error

    def _whitespace_only_ids(self) -> set[int]:
        """Identify standalone whitespace tokens without banning Unicode byte pieces."""
        result: set[int] = set()
        special_set = set(SPECIAL_TOKENS)
        for token, token_id in self.tokenizer.vocab.items():
            if token in special_set:
                continue
            try:
                decoded = self.tokenizer.decode([token_id], skip_special=False, errors="strict")
            except UnicodeDecodeError:
                continue
            if decoded and not decoded.strip():
                result.add(int(token_id))
        return result

    def reset(self) -> None:
        with self._lock:
            self.history.clear()

    def _remember_exchange(self, user_text: str, response: str) -> None:
        encoded = self.tokenizer.encode(
            f"<user> {user_text}\n<assistant> {response}\n<eos>",
            add_special=False,
            allow_special=True,
        )
        self.history = (self.history + encoded)[-self.context_length:]

    def reply(
        self,
        user_text: str,
        *,
        max_new_tokens: int = 40,
        top_k: int = 5,
        top_p: float = 0.8,
        temperature: float = 0.35,
        allow_generative_fallback: bool = False,
    ) -> str:
        cleaned = str(user_text).strip()
        if not cleaned:
            raise ValueError("Message cannot be empty")

        with self._lock:
            response = intent_response(cleaned)
            if response is not None:
                self._remember_exchange(cleaned, response)
                return response

            if self.retriever is not None:
                retrieved = self.retriever.retrieve(cleaned)
                if retrieved is not None:
                    response, _confidence = retrieved
                    self._remember_exchange(cleaned, response)
                    return response

            if not allow_generative_fallback:
                self._remember_exchange(cleaned, SAFE_FALLBACK)
                return SAFE_FALLBACK

            # Match format_conversation exactly:
            # <user> message\n<assistant> response\n<eos>
            message_ids = self.tokenizer.encode(
                " " + cleaned + "\n",
                add_special=False,
                allow_special=False,
            )
            assistant_prefix = self.tokenizer.encode(
                " ",
                add_special=False,
                allow_special=False,
            )
            max_message_tokens = max(1, self.context_length - len(assistant_prefix) - 2)
            message_ids = message_ids[-max_message_tokens:]
            turn_prefix = [self.user_id] + message_ids + [self.assistant_id] + assistant_prefix
            history_budget = max(0, self.context_length - len(turn_prefix))
            prompt = self.history[-history_budget:] + turn_prefix if history_budget else turn_prefix
            generated = self.engine.generate_response(
                prompt,
                max_new_tokens=max_new_tokens,
                eos_id=self.eos_id,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stop_ids={self.user_id, self.assistant_id},
                banned_ids={
                    self.pad_id,
                    self.unk_id,
                    self.bos_id,
                    self.sep_id,
                    self.mask_id,
                    *self.whitespace_only_ids,
                },
                min_new_tokens=2,
                repetition_penalty=1.1,
            )

            visible_ids = list(generated)
            if visible_ids and visible_ids[-1] in {self.user_id, self.assistant_id, self.eos_id}:
                visible_ids = visible_ids[:-1]
            # Arbitrary sampled byte tokens can occasionally form an incomplete
            # UTF-8 sequence. Ignore only those invalid bytes at the UI boundary;
            # tokenizer round-trip tests remain strict.
            response = self.tokenizer.decode(
                visible_ids,
                skip_special=True,
                errors="ignore",
            ).strip()
            if not response:
                token_names = [self.tokenizer.inv_vocab.get(token_id, "<invalid>") for token_id in generated]
                raise RuntimeError(
                    "The model produced no visible text. Generated tokens: "
                    + repr(token_names[:12])
                )

            history_tail = list(generated)
            if history_tail and history_tail[-1] in {self.user_id, self.assistant_id}:
                history_tail.pop()
            if not history_tail or history_tail[-1] != self.eos_id:
                history_tail.append(self.eos_id)
            self.history = (prompt + history_tail)[-self.context_length:]
            return response


__all__ = ["ChatService", "SPECIAL_TOKENS"]
