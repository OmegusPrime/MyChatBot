import numpy as np
from Model.Layers import SinusoidalPositionalEncoding,LayerNormScratch,PositionWiseFeedForward
from Model.Attention import CausalMultiHeadAttention
class TransformerDecoderBlock:
    def __init__(self,embed_dim = 32, num_heads = 4, d_ff=128):
        self.ln1 = LayerNormScratch(embed_dim=embed_dim)
        self.ln2 = LayerNormScratch(embed_dim=embed_dim)
        self.attn = CausalMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = PositionWiseFeedForward(embed_dim=embed_dim,d_ff=d_ff)
    def forward(self,x):
        x_norm1 = self.ln1.forward(x)
        x=x+self.attn.forward(x_norm1)
        x_norm2 = self.ln2.forward(x)
        x=x+self.attn.forward(x_norm2)
        return x
class TransformerCoreStack:
    def __init__(self, vocab_size, embed_dim=32, num_heads=4, d_ff=128, num_blocks=2, max_seq_len=512):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pos_encoder = SinusoidalPositionalEncoding(max_seq_len=max_seq_len, embed_dim=embed_dim)
        self.blocks = [
            TransformerDecoderBlock(embed_dim=embed_dim, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_blocks)
        ]
        self.final_ln = LayerNormScratch(embed_dim=embed_dim)
    def forward(self,embedded_sequence):
        x = self.pos_encoder.forward(embedded_sequence)
        for block in self.blocks:
            x = block.forward(x)
        return self.final_ln.forward(x)
