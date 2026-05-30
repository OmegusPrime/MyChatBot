import numpy as np


class SinusoidalPositionalEncoding:
    def __init__(self, max_seq_len=512, embed_dim=32):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.pe = np.zeros((max_seq_len, embed_dim))
        position = np.arange(0, max_seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        assert seq_len <= self.max_seq_len, f"{seq_len} > {self.max_seq_len}"
        return x + self.pe[:seq_len, :]


class LayerNormScratch:
    def __init__(self, embed_dim=32, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / (np.sqrt(variance + self.eps))
        return self.gamma * x_norm + self.beta


class PositionWiseFeedForward:
    def __init__(self, embed_dim=32, d_ff=128):
        limit1 = np.sqrt(2.0 / (embed_dim + d_ff))
        self.W1 = np.random.normal(0, limit1, (embed_dim, d_ff))
        self.b1 = np.zeros(d_ff)
        limit2 = np.sqrt(2.0 / (d_ff + embed_dim))
        self.W2 = np.random.normal(0, limit2, (d_ff, embed_dim))
        self.b2 = np.zeros(embed_dim)

    def gelu(self, x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward(self, x):
        hidden = self.gelu(np.matmul(x, self.W1) + self.b1)
        return np.matmul(hidden, self.W2) + self.b2
