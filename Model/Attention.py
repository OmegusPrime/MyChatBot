import numpy as np
class CausalMultiHeadAttention:
    def __init__(self, embed_dim =32, num_heads = 4):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        limit = np.sqrt(2.0/(embed_dim+embed_dim))
        self.W_q = np.random.normal(0, limit, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, limit, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, limit, (embed_dim, embed_dim))
        self.W_o = np.random.normal(0, limit, (embed_dim, embed_dim))
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        Q_flat = np.matmul(x, self.W_q)
        K_flat = np.matmul(x, self.W_k)
        V_flat = np.matmul(x, self.W_v)
        Q = Q_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        mask = np.tril(np.ones((seq_len, seq_len)))
        scores = np.where(mask==0, scores-1e9, scores)
        attention_weights = self.softmax(scores)
        context_head = np.matmul(attention_weights, V)
        context_combined = context_head.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        return np.matmul(context_combined, self.W_o)