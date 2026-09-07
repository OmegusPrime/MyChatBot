import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Model.Transformer import TransformerCoreStack
from Pipeline.tokenizer import SPECIAL_TOKENS, Tokenizer
from chat_backend import ChatService
from main_and_eval import train_generative_model


class TestTrainingIntegration(unittest.TestCase):
    def test_tiny_model_trains_saves_loads_and_replies(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tokenizer_path = root / "tokenizer.json"
                checkpoint_path = root / "chatbot_transformer.pt"
                conversation = "<user> hi\n<assistant> hello there\n<eos>\n"

                tokenizer = Tokenizer()
                tokenizer.train([conversation] * 20, vocab_size=300, min_freq=1)
                tokenizer.save(str(tokenizer_path))
                stream = np.asarray(tokenizer.encode(conversation * 80), dtype=np.int32)

                model = TransformerCoreStack(
                    vocab_size=tokenizer.vocab_size,
                    embed_dim=16,
                    num_heads=4,
                    d_ff=32,
                    num_blocks=1,
                    max_seq_len=16,
                )
                train_generative_model(
                    model,
                    stream,
                    stream,
                    tokenizer,
                    checkpoint_path,
                    epochs=8,
                    batch_size=32,
                    context_len=8,
                    learning_rate=0.01,
                    max_samples_per_epoch=256,
                    device=torch.device("cpu"),
                    seed=7,
                    tokenizer_hash=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                )

                self.assertTrue(checkpoint_path.exists())
                service = ChatService(root, device="cpu", backend="legacy")
                try:
                    response = service.reply("hi", max_new_tokens=8, temperature=0)
                    self.assertTrue(response)
                    self.assertNotIn("Ġ", response)
                    self.assertFalse(any(token in response for token in SPECIAL_TOKENS))
                finally:
                    service.close()
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
