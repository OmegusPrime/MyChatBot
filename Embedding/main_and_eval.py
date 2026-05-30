"""
Filename: Embedding/main_and_eval.py
Description: End-to-end modular pipeline coordinating advanced safety-gated Ingest parsing,
             incremental Byte-Pair Encoding (BPE), unigram noise table updates, 
             and custom manual gradient matrix backpropagation.
"""

import os
import sys
import numpy as np

# Resolve workspace root (D:\MyChatBot) so sibling directories can cross-import smoothly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Internal Pipeline & Structural Module Imports ---
from Pipeline.ingest import Ingest  # Advanced document scraper with built-in constraints
from Pipeline.tokenizer import Tokenizer  # Custom GPT-2-shim, stream-trained BPE module

# --- Custom Neural Core Imports ---
from DenseEmbeddingEngine import DenseEmbeddingEngine
from TrainingSampler import TrainingSampler


def calculate_cosine_similarity(vec_a, vec_b):
    """Calculates vector orientation similarity to validate learning progress."""
    dot_prod = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_prod / (norm_a * norm_b)


def execute_end_to_end_pipeline(data_dir, bpe_json_path, export_matrix_path):
    print("=" * 75)
    print("        INITIALIZING ADVANCED CHATBOT EMBEDDING TRAINING ENGINE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # STEP 1: Streaming Ingestion & Dynamic Tokenizer Training
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Constructing token schema and streaming corpus parsing...")

    # Target parameter configuration matching local project capabilities
    target_vocab_capacity = 2500

    # Check if the vocabulary has been compiled or saved yet
    if not os.path.exists(bpe_json_path):
        print(f" -> Presaved configuration not spotted at {bpe_json_path}")
        print(f" -> Training brand new BPE vocabulary using streaming generator...")
        tokenizer = Tokenizer.train_from_ingest(
            data_dir=data_dir,
            vocab_size=target_vocab_capacity,
            save_path=bpe_json_path
        )
    else:
        print(f" -> Found matching vocabulary template. Loading: {bpe_json_path}")
        tokenizer = Tokenizer.from_pretrained(bpe_json_path)

    vocab_size = tokenizer.vocab_size
    print(f" -> Tokenizer active. Managed internal dictionary size: {vocab_size} slots.")

    # Stream the dataset once more to generate a comprehensive, flat numerical ID pipeline
    print(" -> Compiling flat mathematical identifier arrays from ingestion generator...")
    token_id_stream = []

    # Spawn an instance of the production parsing matrix
    ingest_worker = Ingest(path=data_dir)

    # Extract segments across txt, csv, pdf, and multi-turn interaction schemas
    for document_packet in ingest_worker.ingest():
        text_chunk = document_packet.get("text", "")
        if text_chunk.strip():
            # Always encodes directly down to an explicitly flat structural list of integers
            encoded_ids = tokenizer.encode(text_chunk, add_special=False)
            token_id_stream.extend(encoded_ids)

    print(f" -> Extraction complete. Generated array size: {len(token_id_stream):,} elements.")

    # Safety cut-off check to handle insufficient dataset situations gracefully
    if len(token_id_stream) < 5:
        print("\n[CRITICAL ERROR] Generated token stream size is too small to build embeddings.")
        print("Please check that your text, csv, or pdf files are placed correctly in the Dataset directory.")
        return

    # -------------------------------------------------------------------------
    # STEP 2: Allocating Core Structural Elements & Samplers
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Allocating math engines and context noise tables...")

    # Initialize the sampler
    sampler = TrainingSampler(token_ids=token_id_stream, vocab_size=vocab_size, power=0.75)

    # Allocate our customizable neural layers with uniform initialization boundaries
    embedding_dim = 32
    learning_rate = 0.03
    engine = DenseEmbeddingEngine(vocab_size=vocab_size, embedding_dim=embedding_dim, learning_rate=learning_rate)

    # -------------------------------------------------------------------------
    # STEP 3: Continuous Training Pipeline Loop
    # -------------------------------------------------------------------------
    epochs = 12
    window_radius = 3
    negative_draws = 5

    print(f"\n[Step 3/4] Commencing optimization across {epochs} global training iterations...")
    print(
        f" -> Hyperparams: Dim={embedding_dim}, LR={learning_rate}, Window={window_radius}, Negatives={negative_draws}")

    for epoch in range(1, epochs + 1):
        epoch_cross_entropy = 0.0
        step_iterations = 0

        # Pull slide window contexts dynamically from our sampler
        samples = list(sampler.generate_window_samples(window_size=window_radius, num_negatives=negative_draws))

        # Shuffle training pairs every epoch to break structural data patterns
        np.random.shuffle(samples)

        for target_id, pos_context_id, negative_ids in samples:
            # Execute explicit forward-backward SGD step on the dense matrices
            step_loss = engine.optimize_step(target_id, pos_context_id, negative_ids)
            epoch_cross_entropy += step_loss
            step_iterations += 1

        mean_step_loss = epoch_cross_entropy / max(1, step_iterations)

        # Format diagnostic status reports cleanly
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            print(f"   * Epoch [{epoch:02d}/{epochs:02d}] Optimization Complete -> Mean Loss: {mean_step_loss:.5f}")

    # Export our final weight parameters securely to disk
    print(f" -> Training sequences complete. Exporting array matrices to: {export_matrix_path}")
    engine.save_weights(export_matrix_path)

    # -------------------------------------------------------------------------
    # STEP 4: Semantic Matrix Validation & Evaluation Suite
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Executing structural cosine validation suite...")

    # Gather testing parameters out of standard vocabulary listings
    sample_targets = ["data", "system", "python", "backend", "database", "query"]
    available_test_words = [w for w in sample_targets if w in tokenizer.vocab]

    if len(available_test_words) >= 2:
        print("\n--- Calculated Vector Similarities (Post-Optimization Evaluator) ---")
        base_keyword = available_test_words[0]
        base_token_id = tokenizer.vocab[base_keyword]
        vec_base = engine.get_token_vector(base_token_id)

        for variant_keyword in available_test_words[1:]:
            variant_token_id = tokenizer.vocab[variant_keyword]
            vec_variant = engine.get_token_vector(variant_token_id)

            similarity_coefficient = calculate_cosine_similarity(vec_base, vec_variant)
            print(f" -> CosineMatch('{base_keyword}' <-> '{variant_keyword}'): {similarity_coefficient:.5f}")
    else:
        print(" -> System Note: Insufficient custom token keys registered inside the vocab to test pairs.")

    print("\n" + "=" * 75)
    print("             PIPELINE PROCESSING RUN EXECUTED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    # Define production paths relative to your local folder architecture
    DATASET_DIRECTORY = os.path.join(BASE_DIR, "Dataset")
    BPE_CONFIG_PATH = os.path.join(BASE_DIR, "Pipeline", "bpe.json")
    EMBEDDING_EXPORT = os.path.join(BASE_DIR, "Embedding", "trained_chatbot_embeddings.npz")

    # Verify the Dataset folder actually exists before initializing the pipeline
    if not os.path.exists(DATASET_DIRECTORY):
        print(f"[FATAL DIRECTORY ERROR] Could not locate input dataset directory at: {DATASET_DIRECTORY}")
    else:
        execute_end_to_end_pipeline(
            data_dir=DATASET_DIRECTORY,
            bpe_json_path=BPE_CONFIG_PATH,
            export_matrix_path=EMBEDDING_EXPORT
        )