import torch
import torch.nn.functional as F

class ChatBotInferenceEngine:
    def __init__(self, model, context_length=16):
        self.model = model
        self.context_length = context_length

    @torch.no_grad()
    def generate_response(self, initial_token_ids, max_new_tokens=20, eos_id=3, top_k=5, top_p=0.4, temperature=0.7):
        self.model.eval()
        # Strictly slice the initial input to fit your exact pre-trained context length boundary (32)
        working_sequence = list(initial_token_ids[-self.context_length:])
        device = next(self.model.parameters()).device

        for _ in range(max_new_tokens):
            # CRITICAL FIX: Ensure that even as new tokens are appended,
            # we ALWAYS slide the window to only look at the LAST 32 tokens max.
            context_slice = working_sequence[-self.context_length:]
            input_tensor = torch.tensor([context_slice], dtype=torch.long, device=device)

            logits = self.model(input_tensor)
            # Isolate the logits of the absolute last token in our context slice
            next_token_logits = logits[0, -1, :]

            # Apply Temperature scaling to sharpen confident predictions
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply tight top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = float('-inf')

            # Apply top-p (nucleus) filtering
            probs = F.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token smoothly
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            working_sequence.append(next_token_id)
            if next_token_id == eos_id:
                break

        return working_sequence