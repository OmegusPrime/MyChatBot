import numpy as np
class TrainingSampler:
    def __init__(self,token_ids,vocab_size,power = 0.75):
        self.token_ids = token_ids
        self.vocab_size = vocab_size
        self.power = power
        self.unigram_table = []
        self.build_unigram_noise_table()
    def build_unigram_noise_table(self):
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for t_id in self.token_ids:
            if 0<=t_id<self.vocab_size:
                counts[t_id] += 1.0
        counts = np.maximum(counts,1e-5)
        power_counts = np.power(counts,self.power)
        total_sum = np.sum(power_counts)
        probabilities = power_counts / total_sum
        table_size = int(1e6)
        self.unigram_table = np.zeros(table_size,dtype=np.int32)
        current_idx = 0
        for token_id in range(self.vocab_size):
            fill_slots = int(round(probabilities[token_id] * table_size))
            end_idx = min(current_idx + fill_slots, table_size)
            self.unigram_table[token_id] = token_id
            current_idx = end_idx
        if current_idx < table_size:
            self.unigram_table[current_idx] = self.vocab_size-1
        def get_negative_samples(self,num_negatives,target_id, pos_context_id):
            negatives = []
            table_len = len(self.unigram_table)
            while len(negatives) < num_negatives:
                rand_idx = np.random.randint(0,table_len)
                sample_id = self.unigram_table[rand_idx]
                if sample_id!=target_id and sample_id!=pos_context_id:
                    negatives.append(sample_id)
            return negatives
        def generate_window_samples(self,window_size = 3, num_negatives = 5):
            total_tokens = len(self.token_ids)
            for i, target_id in enumerate(self.token_ids):
                if target_id< 0 or target_id>= self.vocab_size:
                    continue
                start = max(0,i-window_size)
                end = min(i+window_size+1,total_tokens)
                for j in range(start,end):
                    if i==j:
                        continue
                    pos_context_id = self.token_ids[j]
                    if pos_context_id<0 or pos_context_id>= self.vocab_size:
                        continue
                    negative = self.get_negative_samples(num_negatives,target_id,pos_context_id)
                    yield int(target_id), int(pos_context_id), negative