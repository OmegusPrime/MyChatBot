import numpy as np
class DenseEmbeddingEngine:
    def __init__(self, vocab_size, embedding_dim=32, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.dim = embedding_dim
        self.lr = learning_rate
        bound = np.sqrt(6.0 / (vocab_size + embedding_dim))
        self.W_target = np.random.uniform(-bound, bound, (vocab_size, embedding_dim))
        self.W_context = np.random.uniform(-bound, bound, (vocab_size, embedding_dim))
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))
    def expand_embedding_matrix(self, new_vocab_size):
        current_rows = self.W_target.shape[0]
        if new_vocab_size <= current_rows:
            return
        needed_rows = new_vocab_size - current_rows
        bound = np.sqrt(6.0 / (new_vocab_size + self.dim))
        new_targets = np.random.uniform(-bound, bound, (needed_rows, self.dim))
        new_contexts = np.random.uniform(-bound, bound, (needed_rows, self.dim))
        self.W_target = np.vstack([self.W_target, new_targets])
        self.W_context = np.vstack([self.W_context, new_contexts])
        self.vocab_size = new_vocab_size
        print(f" -> [Matrix Expanded] Embedding Engine sizes updated to: {self.W_target.shape}")
    def optimize_batch_step(self, b_targets, b_positives, b_negatives):
        v_targets = self.W_target[b_targets]
        u_positives = self.W_context[b_positives]
        scores_pos = np.sum(u_positives * v_targets, axis=1)
        probs_pos = self._sigmoid(scores_pos)
        loss = -np.sum(np.log(probs_pos + 1e-7))
        grad_pos_scalar = (probs_pos - 1.0).reshape(-1, 1)
        grad_wrt_targets = grad_pos_scalar * u_positives
        grad_wrt_positives = grad_pos_scalar * v_targets
        u_negs = self.W_context[b_negatives]
        scores_negs = np.einsum('bik,bk->bi', u_negs, v_targets)
        probs_negs = self._sigmoid(scores_negs)
        loss -= np.sum(np.log(1.0 - probs_negs + 1e-7))
        grad_wrt_targets += np.einsum('bi,bik->bk', probs_negs, u_negs)
        grad_wrt_negs = probs_negs.reshape(probs_negs.shape[0], probs_negs.shape[1], 1) * v_targets.reshape(
            v_targets.shape[0], 1, v_targets.shape[1])
        np.clip(grad_wrt_targets, -1.0, 1.0, out=grad_wrt_targets)
        np.clip(grad_wrt_positives, -1.0, 1.0, out=grad_wrt_positives)
        np.clip(grad_wrt_negs, -1.0, 1.0, out=grad_wrt_negs)
        np.add.at(self.W_target, b_targets, -self.lr * grad_wrt_targets)
        np.add.at(self.W_context, b_positives, -self.lr * grad_wrt_positives)
        np.add.at(self.W_context, b_negatives, -self.lr * grad_wrt_negs)
        return loss / len(b_targets)
    def get_token_vector(self, token_id):
        return (self.W_target[token_id] + self.W_context[token_id]) / 2.0
    def save_weights(self, filepath):
        np.savez(filepath, W_target=self.W_target, W_context=self.W_context)