import torch
import torch.nn as nn
import math


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # FIX #5: Create an absolute causal mask assignment (no subtraction leakage)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_probs, v)

        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(context)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, d_ff)
        self.W2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.W2(self.relu(self.W1(x)))


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff):
        super().__init__()
        self.attn = CausalMultiHeadAttention(embed_dim, num_heads)
        self.ffn = PositionWiseFeedForward(embed_dim, d_ff)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # First residual branch: Attention
        x = x + self.attn(self.ln1(x))
        # FIX #2: Correctly route the second branch through the FFN layer
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerCoreStack(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, d_ff=512, num_blocks=4, max_seq_len=512):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, d_ff) for _ in range(num_blocks)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights (standard language modeling practice)
        self.token_embeddings.weight = self.lm_head.weight

    def forward(self, token_ids):
        if token_ids.dtype != torch.long:
            raise TypeError(f"token_ids must have dtype torch.long, got {token_ids.dtype}")
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape (batch, sequence), got {tuple(token_ids.shape)}")
        B, T = token_ids.shape
        device = token_ids.device
        if T == 0:
            raise ValueError("token_ids cannot contain an empty sequence")
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        if token_ids.numel() and (token_ids.min().item() < 0 or token_ids.max().item() >= self.vocab_size):
            raise ValueError(f"token IDs must be in [0, {self.vocab_size})")

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_embeddings(token_ids) + self.position_embeddings(pos)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))
        return logits

    def config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_blocks": self.num_blocks,
            "max_seq_len": self.max_seq_len,
        }
