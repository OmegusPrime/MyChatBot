import numpy as np
class DenseEmbeddingEngine:
    def __init__(self, vocab_size, embedding_dim = 64, learning_rate = 0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        bound = 1.0/embedding_dim
        self.W_target = np.random.uniform(-bound, bound, (vocab_size, embedding_dim))
        self.W_context = np.random.uniform(-bound, bound, (vocab_size, embedding_dim))
        def sigmoid(self,x):
            clipped_x = np.clip(x, -50.0, 50.0)
            return 1.0/(1.0 + np.exp(-clipped_x))
        def optimize_step(self, target_id, pos_context_id, neg_context_id):
            v_target = self.W_target[target_id]
            u_pos = self.W_context[pos_context_id]
            score_pos = np.dot(v_target, u_pos)
            prob_pos = self.sigmoid(score_pos)
            loss = -np.log(prob_pos+1e-12)
            grad_v_target = (prob_pos-1.0)*u_pos
            grad_v_pos = (prob_pos-1.0)*v_target
            for neg_id in neg_context_id:
                u_neg = self.W_context[neg_id]
                score_neg = np.dot(u_neg, v_target)
                prob_neg = self.sigmoid(score_neg)
                loss_neg = -np.log(prob_neg+1e-12)
                grad_v_target += prob_neg*u_neg
                grad_v_pos+=prob_neg*u_pos
                grad_u_neg = prob_neg*v_target
            self.W_context[pos_context_id] -=self.learning_rate*grad_u_neg
            self.W_target[target_id] -=self.learning_rate*grad_v_target
            return loss
        def get_token_vector(self, token_id):
            if 0<=token_id<self.vocab_size:
                return (self.W_target[token_id]+self.W_context[token_id])/2.0
            else:
                raise IndexError(f"Token ID {token_id} out of vocabulary")
        def save_weights(self,filepath):
            np.savez(filepath, W_target=self.W_target, W_context=self.W_context)
            print(f"Matrix parameters written successfully to disk: {filepath}")
        def load_weights(self,filepath):
            data = np.load(filepath)
            self.W_target = data['W_target']
            self.W_context = data['W_context']
            print(f"Matrix parameters loaded successfully from: {filepath}")