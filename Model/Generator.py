import numpy as np
class LanguageModelHead:
    def __init__(self, embed_dim = 32, vocab_size = 2500):
        limit = np.sqrt(2.0/(embed_dim+vocab_size))
        self.W_out = np.random.normal(0, limit, (embed_dim, vocab_size))
        self.b_out = np.zeros(vocab_size)
    def forward(self, hidden_states):
        return np.matmul(hidden_states, self.W_out) + self.b_out
class ChatBotInferenceEngine:
    def __init__(self, transformer_stack, lm_head, embedding_lookup_func):
        self.model = transformer_stack
        self.lm_head = lm_head
        self.get_vector = embedding_lookup_func
    def softmax(self, logits):
        exp_logits = np.exp(logits-np.max(logits))
        return exp_logits/np.sum(exp_logits)
    def sample_top_k_top_p(self, logits, top_k=50, top_p=0.9):
        probs = self.softmax(logits)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        if top_k > 0:
            sorted_probs[top_k:] = 0.0
        cumulative_probs = np.cumsum(sorted_probs)
        to_remove = cumulative_probs > top_p
        if np.any(to_remove):
            first_cutoff_idx = np.argmax(to_remove)+1
            sorted_probs[first_cutoff_idx:] = 0.0
        if np.sum(sorted_probs)==0.0:
            sorted_probs[:top_k]= 1.0/top_k
        sorted_probs = sorted_probs/np.sum(sorted_probs)
        sample_idx = np.random.choice(sorted_indices, p=sorted_probs)
        return int(sample_idx)
    def  generate_response(self, initial_token_ids, max_nex_tokens = 30, tokenizer_eos_id = 3, top_k =40, top_p = 0.85):
        working_sequence = list(initial_token_ids)
        for _ in range(max_nex_tokens):
            embedded_sequence = np.array([self.get_vector(t_id) for t_id in working_sequence])
            input_tensor = np.expand_dims(embedded_sequence, axis=0)
            hidden_context = self.model.forward(input_tensor)
            logits_tensor = self.lm_head.forward(hidden_context)
            last_token_logits = logits_tensor[0,-1,:]
            next_token_id = self.sample_top_k_top_p(last_token_logits, top_k=top_k, top_p=top_p)
            working_sequence.append(next_token_id)
            if next_token_id == tokenizer_eos_id:
                break
        return working_sequence
