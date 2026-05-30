import numpy as np
class TrainingSampler:
    def __init__(self, token_ids, vocab_size, power=0.75):
        self.token_ids = np.array(token_ids, dtype=np.int32)
        self.vocab_size = vocab_size
        self.power = power
        self.unigram_table = []
        self.build_unigram_noise_table()
    def build_unigram_noise_table(self):
        counts = np.bincount(self.token_ids, minlength=self.vocab_size).astype(np.float64)
        counts = np.maximum(counts, 1e-5)
        power_counts = np.power(counts, self.power)
        probabilities = power_counts / np.sum(power_counts)
        table_size = int(1e6)
        self.unigram_table = np.zeros(table_size, dtype=np.int32)
        current_idx = 0
        for token_id in range(self.vocab_size):
            fill_slots = int(round(probabilities[token_id] * table_size))
            end_idx = min(current_idx + fill_slots, table_size)
            self.unigram_table[current_idx:end_idx] = token_id
            current_idx = end_idx
        if current_idx < table_size:
            self.unigram_table[current_idx:] = self.vocab_size - 1
    def generate_batch_samples(self, window_size=3, num_negatives=5, batch_size=4096):
        total_tokens = len(self.token_ids)
        table_len = len(self.unigram_table)
        targets = []
        positives = []
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue
            if offset > 0:
                t_slice = self.token_ids[:-offset]
                p_slice = self.token_ids[offset:]
            else:
                t_slice = self.token_ids[-offset:]
                p_slice = self.token_ids[:offset]
            targets.append(t_slice)
            positives.append(p_slice)
        all_targets = np.concatenate(targets)
        all_positives = np.concatenate(positives)
        total_samples = len(all_targets)
        for i in range(0, total_samples, batch_size):
            end_i = min(i + batch_size, total_samples)
            curr_batch_size = end_i - i
            b_targets = all_targets[i:end_i]
            b_positives = all_positives[i:end_i]
            rand_indices = np.random.randint(0, table_len, size=(curr_batch_size, num_negatives))
            b_negatives = self.unigram_table[rand_indices]
            yield b_targets, b_positives, b_negatives