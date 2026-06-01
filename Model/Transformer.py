import torch
import torch.nn as nn
import math


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
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
        # FIX #8: Expanded embedding scale for rich semantic capacity
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
        B, T = token_ids.shape
        device = token_ids.device

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_embeddings(token_ids) + self.position_embeddings(pos)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))
        return logits