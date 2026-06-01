
import os
import sys
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Resolve workspace root (D:\MyChatBot) so sibling directories cross-import cleanly
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Internal Pipeline & Structural Module Imports ---
from Pipeline.ingest import Ingest          # Advanced document parser with metadata filtering
from Pipeline.tokenizer import Tokenizer    # Custom GPT-2-shim, stream-trained BPE module

# --- Custom Architecture Core & Generator Head Imports ---
from Model.Transformer import TransformerCoreStack
from Model.Generator import ChatBotInferenceEngine

# --- Configuration Matrix Hyperparameters ---
EMBED_DIM = 128
NUM_HEADS = 4
D_FF = 512
NUM_BLOCKS = 4
CONTEXT_LENGTH = 32
BATCH_SIZE = 64
LM_EPOCHS = 5
LEARNING_RATE = 5e-4

# --- Hardcoded Project Path Environment Bindings ---
DATASET_DIRECTORY = os.path.join(BASE_DIR, "Dataset")
BPE_CONFIG_PATH = os.path.join(BASE_DIR, "Pipeline", "bpe.json")
BINARY_NPY_PATH = os.path.join(BASE_DIR, "Embedding", "token_stream.npy")
MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "Embedding", "chatbot_transformer.pt")


def load_token_stream(data_dir, bpe_path, db_path, npy_path):
    """
    Self-healing cache engine. Automatically detects missing token streams,
    parses raw text via the filtered Ingest layout, and caches a clean binary stream.
    """
    # 1. Initialize or load the trained sub-word BPE tokenizer
    if not os.path.exists(bpe_path):
        print(" -> Presaved BPE config not found. Training vocabulary from ingest...")
        tokenizer = Tokenizer.train_from_ingest(data_dir=data_dir, vocab_size=2500, save_path=bpe_path)
    else:
        print(f" -> Loading Tokenizer from file asset: {bpe_path}")
        tokenizer = Tokenizer.from_pretrained(bpe_path)

    # 2. Check for an active binary token stream cache file on disk
    if os.path.exists(npy_path):
        print(f" -> FAST CACHE HIT: Loading token stream from binary file: {npy_path}")
        token_stream = np.load(npy_path)
        return tokenizer, token_stream

    # 3. SELF-HEALING FALLBACK: Build token stream from scratch if missing
    print(f" -> CACHE MISS: Generating clean token stream array map into {npy_path}...")

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
        print(" -> Parsing raw data source files via metadata-filtered Ingest schema...")
        ingest_worker = Ingest(path=data_dir)
        batch_buffer = []
        for packet in ingest_worker.ingest():
            text_chunk = packet.get("text", "")
            if text_chunk.strip():
                batch_buffer.append((packet.get("file", "unknown"), packet.get("type", "txt"), text_chunk))
                if len(batch_buffer) >= 1000:
                    cursor.executemany("INSERT INTO processed_documents (file_name, doc_type, raw_text) VALUES (?, ?, ?)", batch_buffer)
                    conn.commit()
                    batch_buffer.clear()
        if batch_buffer:
            cursor.executemany("INSERT INTO processed_documents (file_name, doc_type, raw_text) VALUES (?, ?, ?)", batch_buffer)
            conn.commit()

    print(" -> Running BPE tokenizer across database rows to build a clean binary cache layer...")
    token_id_stream = []
    cursor.execute("SELECT raw_text FROM processed_documents")
    for row in cursor.fetchall():
        encoded_ids = tokenizer.encode(row[0], add_special=False)
        token_id_stream.extend(encoded_ids)
    conn.close()

    # Save to disk as an explicit NumPy array map for instantaneous subsequent launches
    token_stream = np.array(token_id_stream, dtype=np.int32)
    np.save(npy_path, token_stream)
    print(f" -> SUCCESS: Clean binary token stream cache written to: {npy_path}")

    return tokenizer, token_stream


def train_generative_model(model, token_stream, epochs, batch_size, context_len, lr, device):
    """
    Executes deep architecture optimization passes utilizing fully vectorized
    PyTorch layers, Cross-Entropy calculations, and gradient clipping safety boundaries.
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_tokens = len(token_stream)
    # Balance processing iterations per epoch for optimal CPU/GPU convergence speeds
    max_samples_per_epoch = min(200000, total_tokens - context_len - 1)

    print(f" -> Commencing model optimization loops across {device} device targets...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0

        # Uniformly sample token positions across the text matrix to balance training context
        start_indices = np.random.randint(0, total_tokens - context_len - 1, size=max_samples_per_epoch)

        for i in range(0, len(start_indices), batch_size):
            batch_starts = start_indices[i : i + batch_size]
            if len(batch_starts) < batch_size:
                continue

            X_list, Y_list = [], []
            for start in batch_starts:
                X_list.append(token_stream[start : start + context_len])
                Y_list.append(token_stream[start + 1 : start + context_len + 1])

            X = torch.tensor(np.array(X_list), dtype=torch.long, device=device)
            Y = torch.tensor(np.array(Y_list), dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(X)

            # Reshape inputs to evaluate classification metrics across the flattened vocab sequence
            loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
            loss.backward()

            # Gradient clipping boundary to maintain parameter variance stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"   * Epoch [{epoch:02d}/{epochs:02d}] Completed -> Mean Training Loss: {total_loss / max(1, steps):.4f}")

    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    print(f" -> Saved active structural transformer checkpoint file to: {MODEL_CHECKPOINT_PATH}")


def run_interactive_terminal_chat(tokenizer, model, device):
    """
    Orchestrates an interactive, multi-turn dialogue loop. Directly matches training IDs
    without data-remapping latency or database connection bottlenecks.
    """
    inference_engine = ChatBotInferenceEngine(model, context_length=CONTEXT_LENGTH)

    print("\n" + "=" * 80)
    print(" >>> SYSTEM GENERATIVE CHATBOT CONVERSATION ENGINE ONLINE <<<")
    print(" Instructions: Type your message and press Enter. Type 'END' to exit safely.")
    print("=" * 80 + "\n")

    conversation_history = []

    while True:
        user_query = input("You: ").strip()
        if user_query.upper() == "END":
            print("\nShutting down interactive session parameters. Goodbye!")
            break
        if not user_query:
            continue

        # Tokenize directly to BPE integer keys matching our parameter shapes
        encoded_input_ids = tokenizer.encode(user_query, add_special=False)
        conversation_history.extend(encoded_input_ids)

        # Cap sliding attention history window length to stay within maximum block context bounds
        if len(conversation_history) > CONTEXT_LENGTH:
            conversation_history = conversation_history[-CONTEXT_LENGTH:]

        output_ids = inference_engine.generate_response(
            initial_token_ids=conversation_history,
            max_new_tokens=25,
            eos_id=3,
            top_k=40,
            top_p=0.85
        )

        newly_generated_tokens = output_ids[len(conversation_history):]
        conversation_history.extend(newly_generated_tokens)

        # Safe decode output while stripping structural structural symbols
        response_text = tokenizer.decode(newly_generated_tokens, skip_special=True)
        if not response_text.strip():
            response_text = "..."

        print(f"Chatbot: {response_text}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SQLITE_DB_PATH = os.path.join(BASE_DIR, "Pipeline", "chatbot_cache.db")

    # Force complete ingestion rebuild if the binary cache file was deleted
    if os.path.exists(SQLITE_DB_PATH) and not os.path.exists(BINARY_NPY_PATH):
        print(" -> Old staging cache database found with un-filtered entries. Purging for clean build...")
        try:
            os.remove(SQLITE_DB_PATH)
        except OSError:
            pass

    print("Checking system data cache state dependencies...")
    tokenizer, token_stream = load_token_stream(
        data_dir=DATASET_DIRECTORY,
        bpe_path=BPE_CONFIG_PATH,
        db_path=SQLITE_DB_PATH,
        npy_path=BINARY_NPY_PATH
    )

    vocab_size = tokenizer.vocab_size
    print(f" -> Loaded vocabulary capacity layout: {vocab_size} tokens.")
    print(f" -> Active stream sequence configuration volume: {len(token_stream):,} tokens.")

    # Initialize the structural network core
    model = TransformerCoreStack(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_blocks=NUM_BLOCKS,
        max_seq_len=512
    ).to(device)

    # Load model state from checkpoint if available, otherwise run training
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f" -> Found existing model checkpoint asset at {MODEL_CHECKPOINT_PATH}. Loading weights...")
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    else:
        print("\n -> Training transformer parameters from a clean initialized state...")
        train_generative_model(model, token_stream, LM_EPOCHS, BATCH_SIZE, CONTEXT_LENGTH, LEARNING_RATE, device)

    run_interactive_terminal_chat(tokenizer, model, device)


if __name__ == "__main__":
    main()