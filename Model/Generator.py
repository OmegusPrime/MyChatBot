"""Autoregressive text generation for the PyTorch transformer."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


class ChatBotInferenceEngine:
    def __init__(self, model, context_length: int = 32):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        model_limit = getattr(model, "max_seq_len", context_length)
        if context_length > model_limit:
            raise ValueError(
                f"context_length {context_length} exceeds model max_seq_len {model_limit}"
            )
        self.model = model
        self.context_length = context_length

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.numel():
            return logits
        threshold = torch.topk(logits, top_k).values[-1]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = logits.clone()
        filtered[sorted_indices[remove]] = float("-inf")
        return filtered

    @torch.no_grad()
    def generate_response(
        self,
        initial_token_ids: Iterable[int],
        max_new_tokens: int = 40,
        eos_id: int = 3,
        top_k: int = 20,
        top_p: float = 0.9,
        temperature: float = 0.8,
        *,
        stop_ids: Iterable[int] | None = None,
        banned_ids: Iterable[int] | None = None,
        min_new_tokens: int = 1,
        repetition_penalty: float = 1.1,
    ) -> list[int]:
        """Return newly generated token IDs, never the prompt prefix."""
        if max_new_tokens <= 0:
            return []
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens must be between 0 and max_new_tokens")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least 1")

        prompt = [int(token_id) for token_id in initial_token_ids]
        if not prompt:
            raise ValueError("initial_token_ids cannot be empty")

        self.model.eval()
        device = next(self.model.parameters()).device
        working_sequence = prompt[-self.context_length:]
        generated: list[int] = []
        stop_set = {int(eos_id), *(int(value) for value in (stop_ids or []))}
        banned_set = {int(value) for value in (banned_ids or [])}

        for step in range(max_new_tokens):
            context_slice = working_sequence[-self.context_length:]
            input_tensor = torch.tensor([context_slice], dtype=torch.long, device=device)
            next_logits = self.model(input_tensor)[0, -1, :].float().clone()

            if repetition_penalty > 1:
                for token_id in set(generated):
                    if 0 <= token_id < next_logits.numel():
                        value = next_logits[token_id]
                        next_logits[token_id] = (
                            value * repetition_penalty if value < 0 else value / repetition_penalty
                        )

            for token_id in banned_set:
                if 0 <= token_id < next_logits.numel():
                    next_logits[token_id] = float("-inf")
            if step < min_new_tokens:
                for token_id in stop_set:
                    if 0 <= token_id < next_logits.numel():
                        next_logits[token_id] = float("-inf")

            if temperature == 0:
                next_token_id = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / temperature
                next_logits = self._apply_top_k(next_logits, top_k)
                next_logits = self._apply_top_p(next_logits, top_p)
                probabilities = F.softmax(next_logits, dim=-1)
                if not torch.isfinite(probabilities).all() or probabilities.sum().item() <= 0:
                    raise RuntimeError("Sampling filters removed every valid output token")
                next_token_id = int(torch.multinomial(probabilities, num_samples=1).item())

            generated.append(next_token_id)
            working_sequence.append(next_token_id)
            if next_token_id in stop_set and len(generated) >= min_new_tokens:
                break

        return generated
