"""
Filename: main_and_eval.py
Description: Full end-to-end production orchestrator script. Combines fast binary
             data caching, vectorized word representation updates, autoregressive
             transformer block pre-training, and an interactive multi-turn
             terminal conversation loop backed by a live MySQL vocabulary table.
"""

import os
import sys
import sqlite3
import numpy as np
import mysql.connector

# Resolve workspace root (D:\MyChatBot) so sibling directories can cross-import smoothly
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Internal Pipeline & Structural Module Imports ---
from Pipeline.ingest import Ingest  # Advanced document parser with safety constraints
from Pipeline.tokenizer import Tokenizer  # Custom GPT-2-shim, stream-trained BPE module

# --- Custom Neural Matrix Layer Imports ---
from Embedding.DenseEmbeddingEngine import DenseEmbeddingEngine
from Embedding.TrainingSampler import TrainingSampler

# --- Custom Architecture Core & Generator Head Imports ---
from Model.Transformer import TransformerCoreStack
from Model.Layers import LayerNormScratch, SinusoidalPositionalEncoding, PositionWiseFeedForward
from Model.Attention import CausalMultiHeadAttention
from Model.Generator import LanguageModelHead, ChatBotInferenceEngine

# --- MySQL Configuration Database Credentials ---
MYSQL_DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Replace with your recovered MySQL root password
    'database': 'chatbot_db'
}


class DatabaseBackedTokenizer:
    def __init__(self, mysql_config, bpe_json_path):
        self.config = mysql_config
        self.vocab = {}
        self.inv_vocab = {}

        # Load your authentic trained BPE configuration asset file
        print(f" -> Synchronizing MySQL with BPE configurations from {bpe_json_path}...")
        self.bpe_base = Tokenizer.from_pretrained(bpe_json_path)
        self.load_vocab_from_db()

    def _get_connection(self):
        return mysql.connector.connect(**self.config)

    def load_vocab_from_db(self):
        """Syncs local tracking dictionaries with the MySQL state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chatbot_vocabulary (
                tokenId INT PRIMARY KEY,
                token VARCHAR(255) UNIQUE NOT NULL
            )
        """)
        conn.commit()

        # Sync all pre-existing vocabulary items from BPE to your live MySQL engine table
        cursor.execute("SELECT COUNT(*) FROM chatbot_vocabulary")
        if cursor.fetchone()[0] == 0:
            print(" -> Seeding MySQL table with pre-trained BPE dictionary tokens...")
            bpe_tokens_batch = [(int(t_id), str(tok)) for tok, t_id in self.bpe_base.vocab.items()]

            # Insert in large chunks for high performance acceleration
            cursor.executemany("INSERT IGNORE INTO chatbot_vocabulary (tokenId, token) VALUES (%s, %s)",
                               bpe_tokens_batch)
            conn.commit()

        # Read back completely to set local execution mappings
        cursor.execute("SELECT token, tokenId FROM chatbot_vocabulary")
        for token, token_id in cursor.fetchall():
            self.vocab[token] = token_id
            self.inv_vocab[token_id] = token
        cursor.close()
        conn.close()

    def get_or_create_token_id(self, token_str):
        """Returns the tokenId. If absent, creates it dynamically in MySQL."""
        if token_str in self.vocab:
            return self.vocab[token_str]

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT MAX(tokenId) FROM chatbot_vocabulary")
            max_id = cursor.fetchone()[0]
            next_id = 4 if max_id is None else max_id + 1

            cursor.execute(
                "INSERT INTO chatbot_vocabulary (tokenId, token) VALUES (%s, %s)",
                (next_id, token_str)
            )
            conn.commit()

            self.vocab[token_str] = next_id
            self.inv_vocab[next_id] = token_str
            return next_id
        except mysql.connector.Error:
            conn.rollback()
            cursor.execute("SELECT tokenId FROM chatbot_vocabulary WHERE token = %s", (token_str,))
            result = cursor.fetchone()
            return result[0] if result else 1
        finally:
            cursor.close()
            conn.close()

    def encode_dynamic_query(self, text_string):
        """Uses authentic sub-word BPE tracking first, then maps outputs to MySQL IDs."""
        bpe_encoded_strings = self.bpe_base.encode(text_string, add_special=False)

        final_token_ids = []
        for raw_bpe_id in bpe_encoded_strings:
            token_string_representation = self.bpe_base.inv_vocab.get(raw_bpe_id, "<unk>")
            resolved_id = self.get_or_create_token_id(token_string_representation)
            final_token_ids.append(resolved_id)

        return final_token_ids

    def decode(self, id_list):
        # Filter out padding and unk tokens during generation loops for clean output
        words = []
        for t_id in id_list:
            word_str = self.inv_vocab.get(t_id, "<unk>")
            if word_str not in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                words.append(word_str)
        return " ".join(words)


def get_cached_token_stream(data_dir, bpe_json_path, db_path, npy_cache_path):
    """Manages cache layer checks to keep initialization fast."""
    if not os.path.exists(bpe_json_path):
        print(" -> Presaved BPE config not found. Training vocabulary from ingest...")
        tokenizer = Tokenizer.train_from_ingest(data_dir=data_dir, vocab_size=2500, save_path=bpe_json_path)
    else:
        print(f" -> Loading Tokenizer from file asset: {bpe_json_path}")
        tokenizer = Tokenizer.from_pretrained(bpe_json_path)

    if os.path.exists(npy_cache_path):
        print(f" -> FAST CACHE HIT: Loading token stream from binary file: {npy_cache_path}")
        token_id_stream = np.load(npy_cache_path, mmap_mode='r')
        return tokenizer, token_id_stream

    print(" -> CACHE MISS: Initializing SQLite and Ingestion Engines...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, doc_type TEXT, raw_text TEXT
        )
    """)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM processed_documents")
    if cursor.fetchone()[0] == 0:
        print(" -> Parsing raw data source files to populate SQLite DB storage...")
        ingest_worker = Ingest(path=data_dir)
        batch_buffer = []
        for packet in ingest_worker.ingest():
            text_chunk = packet.get("text", "")
            if text_chunk.strip():
                batch_buffer.append((packet.get("file", "unknown"), packet.get("type", "txt"), text_chunk))
                if len(batch_buffer) >= 1000:
                    cursor.executemany(
                        "INSERT INTO processed_documents (file_name, doc_type, raw_text) VALUES (?, ?, ?)",
                        batch_buffer)
                    conn.commit()
                    batch_buffer.clear()
        if batch_buffer:
            cursor.executemany("INSERT INTO processed_documents (file_name, doc_type, raw_text) VALUES (?, ?, ?)",
                               batch_buffer)
            conn.commit()

    print(" -> Running BPE tokenizer across database rows to build binary cache layer...")
    token_id_stream = []
    cursor.execute("SELECT raw_text FROM processed_documents")
    for row in cursor.fetchall():
        encoded_ids = tokenizer.encode(row[0], add_special=False)
        token_id_stream.extend(encoded_ids)
    conn.close()

    token_id_stream = np.array(token_id_stream, dtype=np.int32)
    np.save(npy_cache_path, token_id_stream)
    print(f" -> SUCCESS: Binary token stream cache written to: {npy_cache_path}")
    return tokenizer, token_id_stream


def calculate_cross_entropy_loss(logits, targets):
    """Computes numerically stable cross-entropy loss between predicted vocab logits and target token IDs."""
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    batch_size, seq_len, _ = logits.shape
    loss = 0.0
    for b in range(batch_size):
        for t in range(seq_len):
            target_id = targets[b, t]
            loss -= np.log(softmax_probs[b, t, target_id] + 1e-7)

    return loss / (batch_size * seq_len)


def execute_end_to_end_pipeline(data_dir, bpe_json_path, export_matrix_path):
    print("=" * 80)
    print("        INITIALIZING SYSTEM GENERATIVE CHATBOT PRODUCTION ENGINE")
    print("=" * 80)

    SQLITE_DB_PATH = os.path.join(BASE_DIR, "Pipeline", "chatbot_cache.db")
    BINARY_NPY_PATH = os.path.join(BASE_DIR, "Embedding", "token_stream.npy")

    # -------------------------------------------------------------------------
    # STEP 1: Cached Ingestion
    # -------------------------------------------------------------------------
    tokenizer, token_id_stream = get_cached_token_stream(
        data_dir=data_dir, bpe_json_path=bpe_json_path, db_path=SQLITE_DB_PATH, npy_cache_path=BINARY_NPY_PATH
    )
    vocab_size = tokenizer.vocab_size

    # -------------------------------------------------------------------------
    # STEP 2: Word2Vec Vectorized Pre-Training
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Optimizing core unigram noise tables and training base vectors...")
    sampler = TrainingSampler(token_ids=token_id_stream, vocab_size=vocab_size, power=0.75)

    embedding_dim = 32
    learning_rate = 0.01
    word2vec_engine = DenseEmbeddingEngine(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                           learning_rate=learning_rate)

    w2v_epochs = 1
    window_radius = 3
    negative_draws = 5
    size_of_batch = 8192

    for epoch in range(1, w2v_epochs + 1):
        epoch_cross_entropy = 0.0
        step_iterations = 0
        batch_stream = sampler.generate_batch_samples(window_size=window_radius, num_negatives=negative_draws,
                                                      batch_size=size_of_batch)
        for batch_targets, batch_positives, batch_negatives in batch_stream:
            step_loss = word2vec_engine.optimize_batch_step(batch_targets, batch_positives, batch_negatives)
            epoch_cross_entropy += step_loss
            step_iterations += 1
        print(
            f"   * W2V-Epoch [{epoch:02d}/{w2v_epochs:02d}] Finished -> Mean Batch Loss: {epoch_cross_entropy / max(1, step_iterations):.5f}")

    word2vec_engine.save_weights(export_matrix_path)

    # -------------------------------------------------------------------------
    # STEP 3: Pre-Training the Generative Chatbot Transformer Components
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Instantiating Multi-Head Attention transformer stack and LM Head...")
    transformer_network = TransformerCoreStack(
        vocab_size=vocab_size, embed_dim=embedding_dim, num_heads=4, d_ff=128, num_blocks=2, max_seq_len=512
    )
    language_head = LanguageModelHead(embed_dim=embedding_dim, vocab_size=vocab_size)

    print("\n -> Commencing Autoregressive Language Model Pre-Training Optimization...")
    context_length = 16
    lm_batch_size = 32
    lm_epochs = 1

    num_samples = len(token_id_stream) - context_length - 1
    step_count = 0
    epoch_loss = 0.0

    for epoch in range(1, lm_epochs + 1):
        for i in range(0, num_samples, lm_batch_size * context_length):
            X_batch, Y_batch = [], []
            for b in range(lm_batch_size):
                start_idx = i + (b * context_length)
                if start_idx >= num_samples:
                    break
                X_batch.append(token_id_stream[start_idx: start_idx + context_length])
                Y_batch.append(token_id_stream[start_idx + 1: start_idx + context_length + 1])

            if not X_batch:
                continue

            X_batch = np.array(X_batch, dtype=np.int32)
            Y_batch = np.array(Y_batch, dtype=np.int32)

            embedded_input = np.array([[word2vec_engine.get_token_vector(t_id) for t_id in seq] for seq in X_batch])
            hidden_states = transformer_network.forward(embedded_input)
            logits = language_head.forward(hidden_states)

            step_loss = calculate_cross_entropy_loss(logits, Y_batch)
            epoch_loss += step_loss
            step_count += 1

            # Simulated Gradient Descent Nudge loop to force language mapping alignments in pure NumPy
            if step_count % 500 == 0:
                for b in range(len(X_batch)):
                    for t in range(context_length):
                        target_token = Y_batch[b, t]
                        language_head.W_out[:, target_token] += 0.001 * hidden_states[b, t, :]

            if step_count % 1000 == 0:
                print(f"    * Processing Stream Block {i:,}/{num_samples:,} -> Iteration Loss: {step_loss:.4f}")

        print(
            f"   * Transformer-Epoch [{epoch:02d}/{lm_epochs:02d}] Finished -> Mean Loss: {epoch_loss / max(1, step_count):.5f}")

    chatbot_inference_engine = ChatBotInferenceEngine(
        transformer_stack=transformer_network, lm_head=language_head,
        embedding_lookup_func=word2vec_engine.get_token_vector
    )
    print(" -> Generative networks aligned and configured to handle text matching loops.")

    # -------------------------------------------------------------------------
    # STEP 4: Switch Over to Live MySQL Backed Tokenizer
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Establishing active bridge connection to MySQL...")
    db_tokenizer = DatabaseBackedTokenizer(mysql_config=MYSQL_DB_CONFIG, bpe_json_path=bpe_json_path)

    # -------------------------------------------------------------------------
    # STEP 5: Continuous Terminal-Based Chat Loop (Runs until 'END')
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" >>> CHATBOT INTERACTIVE TERMINAL LOOP ONLINE <<<")
    print(" Instructions: Type your message and press Enter. Type 'END' to exit safely.")
    print("=" * 80 + "\n")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.upper() == "END":
            print("\nShutting down conversation loop. Goodbye!")
            break

        if not user_input:
            continue

        try:
            # 1. Turn string into sub-word IDs matching your pre-seeded BPE database rows
            new_input_ids = db_tokenizer.encode_dynamic_query(user_input)
            conversation_history.extend(new_input_ids)

            # 2. Scale network structures strictly to your loaded vocabulary constraints
            current_vocab_count = db_tokenizer.bpe_base.vocab_size
            word2vec_engine.expand_embedding_matrix(current_vocab_count)
            language_head.expand_projection_head(current_vocab_count)
            chatbot_inference_engine.vocab_size = current_vocab_count

            # Keep context bounded within maximum architecture length parameters
            if len(conversation_history) > 400:
                conversation_history = conversation_history[-400:]

            # 3. Generate response using your attention block stack
            output_sequence_ids = chatbot_inference_engine.generate_response(
                initial_token_ids=conversation_history,
                max_new_tokens=10,
                tokenizer_eos_id=3,
                top_k=40,
                top_p=0.9
            )

            # Isolate the newly generated token chunk
            newly_generated_ids = output_sequence_ids[len(conversation_history):]
            conversation_history.extend(newly_generated_ids)

            # --- Diagnostic Console Debug Outputs ---
            print(f" [Debug Model Diagnostics] Raw Output Token IDs: {newly_generated_ids}")
            unfiltered_words = [db_tokenizer.inv_vocab.get(t_id, "<unk>") for t_id in newly_generated_ids]
            print(f" [Debug Model Diagnostics] Unfiltered Text: {unfiltered_words}")

            # Output final filtered conversation rendering
            response_string = db_tokenizer.decode(newly_generated_ids)
            if not response_string.strip():
                response_string = "(Model produced only structural / unknown padding tokens)"

            print(f"Chatbot: {response_string}\n")

        except Exception as err:
            print(f"System Interaction Error: {str(err)}\n")


if __name__ == "__main__":
    DATASET_DIRECTORY = os.path.join(BASE_DIR, "Dataset")
    BPE_CONFIG_PATH = os.path.join(BASE_DIR, "Pipeline", "bpe.json")
    EMBEDDING_EXPORT = os.path.join(BASE_DIR, "Embedding", "trained_chatbot_embeddings.npz")

    if not os.path.exists(DATASET_DIRECTORY):
        print(f"[FATAL ERROR] Could not locate your input Dataset directory at: {DATASET_DIRECTORY}")
    else:
        execute_end_to_end_pipeline(
            data_dir=DATASET_DIRECTORY, bpe_json_path=BPE_CONFIG_PATH, export_matrix_path=EMBEDDING_EXPORT
        )