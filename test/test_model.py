import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TestTransformerAndGeneration(unittest.TestCase):
    def test_forward_shape_and_invariants(self):
        from Model.Transformer import TransformerCoreStack

        model = TransformerCoreStack(
            vocab_size=32,
            embed_dim=16,
            num_heads=4,
            d_ff=32,
            num_blocks=1,
            max_seq_len=8,
        )
        logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 3, 32))
        with self.assertRaises(ValueError):
            TransformerCoreStack(vocab_size=32, embed_dim=15, num_heads=4)
        with self.assertRaises(ValueError):
            model(torch.tensor([[32]], dtype=torch.long))

    def test_generator_returns_only_new_ids_and_stops_at_eos(self):
        from Model.Generator import ChatBotInferenceEngine

        class EosModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.max_seq_len = 8

            def forward(self, token_ids):
                batch, sequence = token_ids.shape
                logits = torch.zeros(batch, sequence, 6, device=token_ids.device)
                logits[..., 3] = 10
                return logits

        engine = ChatBotInferenceEngine(EosModel(), context_length=8)
        generated = engine.generate_response(
            [4, 5],
            max_new_tokens=4,
            eos_id=3,
            temperature=0,
            min_new_tokens=0,
        )
        self.assertEqual(generated, [3])

    def test_checkpoint_loads_through_shared_chat_service(self):
        from Model.Transformer import TransformerCoreStack
        from Pipeline.tokenizer import Tokenizer
        from chat_backend import ChatService

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer.json"
            checkpoint_path = root / "chatbot_transformer.pt"
            tokenizer = Tokenizer()
            tokenizer.train(["hello world"], vocab_size=280, min_freq=1)
            tokenizer.save(str(tokenizer_path))
            model = TransformerCoreStack(
                vocab_size=tokenizer.vocab_size,
                embed_dim=16,
                num_heads=4,
                d_ff=32,
                num_blocks=1,
                max_seq_len=8,
            )
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "model_config": model.config(),
                    "context_length": 8,
                    "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                    "vocab_size": tokenizer.vocab_size,
                },
                checkpoint_path,
            )
            service = ChatService(root, device="cpu")
            self.assertEqual(service.tokenizer.vocab_size, tokenizer.vocab_size)
            self.assertEqual(service.reply("Hi"), "Hi! How can I help you today?")
            self.assertIn("rephrase", service.reply("quantum zebra flux capacitor").lower())
            service.history.extend([1, 2, 3])
            service.reset()
            self.assertEqual(service.history, [])


if __name__ == "__main__":
    unittest.main()
