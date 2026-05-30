import os
import sys
import numpy as np

# Resolve workspace root (D:\MyChatBot) so sibling directories can cross-import smoothly
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Internal Pipeline & Structural Module Imports ---
from Pipeline.ingest import Ingest          # Advanced document parser with safety constraints
from Pipeline.tokenizer import Tokenizer    # Custom GPT-2-shim, stream-trained BPE module

# --- Custom Neural Matrix Layer Imports ---
from Embedding.DenseEmbeddingEngine import DenseEmbeddingEngine
from Embedding.TrainingSampler import TrainingSampler

# --- Custom Architecture Core & Generator Head Imports ---
from Model.Transformer import TransformerCoreStack
from Model.Layers import LayerNormScratch, SinusoidalPositionalEncoding, PositionWiseFeedForward
from Model.Attention import CausalMultiHeadAttention
from Model.Generator import LanguageModelHead, ChatBotInferenceEngine


def calculate_cosine_similarity(vec_a, vec_b):
    """Calculates vector orientation similarity to validate word representation learning."""
    dot_prod = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_prod / (norm_a * norm_b)


def execute_end_to_end_pipeline(data_dir, bpe_json_path, export_matrix_path):
    print("=" * 80)
    print("        INITIALIZING SYSTEM GENERATIVE CHATBOT PRODUCTION ENGINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Streaming Ingestion & Dynamic Tokenizer Training
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Running ingestion mechanics and training BPE vocabulary...")

    target_vocab_capacity = 2500  # Set based on custom training corpus constraints

    # Check if the vocabulary has been compiled or saved yet
    if not os.path.exists(bpe_json_path):
        print(f" -> Presaved BPE config not found at: {bpe_json_path}")
        print(f" -> Training new BPE token dictionary using streaming data generators...")
        tokenizer = Tokenizer.train_from_ingest(
            data_dir=data_dir,
            vocab_size=target_vocab_capacity,
            save_path=bpe_json_path
        )
    else:
        print(f" -> Found matching vocabulary configurations. Loading: {bpe_json_path}")
        tokenizer = Tokenizer.from_pretrained(bpe_json_path)

    vocab_size = tokenizer.vocab_size
    print(f" -> Tokenizer active. Managed dictionary space: {vocab_size} distinct slots.")

    # Stream the dataset again to build a flat list of text Token IDs
    print(" -> Translating text documents into structural identifier streams...")
    token_id_stream = []
    ingest_worker = Ingest(path=data_dir)

    # Stream from the parser across all types (txt, csv, pdf, dailydialog, personachat)
    for document_packet in ingest_worker.ingest():
        text_chunk = document_packet.get("text", "")
        if text_chunk.strip():
            encoded_ids = tokenizer.encode(text_chunk, add_special=False)
            token_id_stream.extend(encoded_ids)

    print(f" -> Extraction complete. Total token sequence volume: {len(token_id_stream):,} tokens.")

    # Safety breakout check for small datasets
    if len(token_id_stream) < 10:
        print("\n[CRITICAL ERROR] Generated token stream size is insufficient to train vectors.")
        print("Please ensure valid chat dialog text data is present in your Dataset folder.")
        return

    # -------------------------------------------------------------------------
    # STEP 2: Pre-Training Base Dense Word Vectors (Word2Vec) - MEMORY SAFE
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Optimizing core unigram noise tables and training base vectors...")

    # Initialize our vectorized sampler
    sampler = TrainingSampler(token_ids=token_id_stream, vocab_size=vocab_size, power=0.75)

    embedding_dim = 32
    learning_rate = 0.001
    word2vec_engine = DenseEmbeddingEngine(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                           learning_rate=learning_rate)

    w2v_epochs = 3  # Reduced epochs since vectorization yields cleaner gradient steps
    window_radius = 3
    negative_draws = 5
    size_of_batch = 8192  # Large batch sizes leverage underlying CPU core optimizations

    print(f" -> Training baseline representations across {w2v_epochs} vectorized iterations...")
    for epoch in range(1, w2v_epochs + 1):
        epoch_cross_entropy = 0.0
        step_iterations = 0

        # Pull high-speed array blocks out of our batch generator
        batch_stream = sampler.generate_batch_samples(
            window_size=window_radius,
            num_negatives=negative_draws,
            batch_size=size_of_batch
        )

        for batch_targets, batch_positives, batch_negatives in batch_stream:
            # Process thousands of updates in a single execution step
            step_loss = word2vec_engine.optimize_batch_step(batch_targets, batch_positives, batch_negatives)
            epoch_cross_entropy += step_loss
            step_iterations += 1

        mean_step_loss = epoch_cross_entropy / max(1, step_iterations)
        print(f"   * W2V-Epoch [{epoch:02d}/{w2v_epochs:02d}] Finished -> Mean Batch Loss: {mean_step_loss:.5f}")

    print(f" -> Base training complete. Exporting optimized weights array to: {export_matrix_path}")
    word2vec_engine.save_weights(export_matrix_path)

    # -------------------------------------------------------------------------
    # STEP 3: Initializing Generative Chatbot Neural Components
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Instantiating Multi-Head Attention transformer stack and LM Head...")

    # Build your custom structural transformer stack elements
    transformer_network = TransformerCoreStack(
        vocab_size=vocab_size,
        embed_dim=embedding_dim,
        num_heads=4,
        d_ff=128,
        num_blocks=2,
        max_seq_len=512
    )

    # Create the linear language head projection mapping back to the dictionary size
    language_head = LanguageModelHead(embed_dim=embedding_dim, vocab_size=vocab_size)

    # Connect everything into your high-level generation manager
    chatbot_inference_engine = ChatBotInferenceEngine(
        transformer_stack=transformer_network,
        lm_head=language_head,
        embedding_lookup_func=word2vec_engine.get_token_vector
    )
    print(" -> All networks integrated successfully. Models configured to handle context mixing.")

    # -------------------------------------------------------------------------
    # STEP 4: Running Basic Vector Cosine Evaluations
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Executing semantic distance evaluations...")

    sample_targets = ["data", "system", "python", "backend", "database", "query"]
    available_test_words = [w for w in sample_targets if w in tokenizer.vocab]

    if len(available_test_words) >= 2:
        print("\n--- Core Matrix Word Distance Check ---")
        base_keyword = available_test_words[0]
        base_token_id = tokenizer.vocab[base_keyword]
        vec_base = word2vec_engine.get_token_vector(base_token_id)

        for variant_keyword in available_test_words[1:]:
            variant_token_id = tokenizer.vocab[variant_keyword]
            vec_variant = word2vec_engine.get_token_vector(variant_token_id)

            similarity_coefficient = calculate_cosine_similarity(vec_base, vec_variant)
            print(f" -> CosSim('{base_keyword}' <-> '{variant_keyword}'): {similarity_coefficient:.5f}")
    else:
        print(" -> System Note: Insufficient overlapping tokens found in vocabulary text to perform tests.")

    # -------------------------------------------------------------------------
    # STEP 5: Interactive Conversational Generation Testing
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Testing autoregressive generation loop outputs...")

    # Select testing tokens present in your vocabulary dictionary
    test_phrase = "python backend database"
    input_ids = tokenizer.encode(test_phrase, add_special=False)

    print(f" -> Input Prompt String: '{test_phrase}'")
    print(f" -> Tokenized Translation: {input_ids}")
    print(" -> Generating autoregressive response matrix using Top-K and Nucleus (Top-P) sampling...")

    output_sequence_ids = chatbot_inference_engine.generate_response(
        initial_token_ids=input_ids,
        max_nex_tokens=12,
        tokenizer_eos_id=3,  # Matches standard EOS definitions inside your tokenizer
        top_k=40,
        top_p=0.85
    )

    print(f" -> Generation complete. Full output tokens sequence: {output_sequence_ids}")
    decoded_response = tokenizer.decode(output_sequence_ids, skip_special=True)
    print(f"\n >>> CHATBOT GENERATION: '{decoded_response}'")

    print("\n" + "=" * 80)
    print("               CONVERSATION WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    # Define system configurations matching your workspace paths
    DATASET_DIRECTORY = os.path.join(BASE_DIR, "Dataset")
    BPE_CONFIG_PATH = os.path.join(BASE_DIR, "Pipeline", "bpe.json")
    EMBEDDING_EXPORT = os.path.join(BASE_DIR, "Embedding", "trained_chatbot_embeddings.npz")

    # Verify input conditions are met before executing the pipeline
    if not os.path.exists(DATASET_DIRECTORY):
        print(f"[FATAL ERROR] Could not locate your input Dataset directory at: {DATASET_DIRECTORY}")
        print("Please verify your file structure layout and try again.")
    else:
        execute_end_to_end_pipeline(
            data_dir=DATASET_DIRECTORY,
            bpe_json_path=BPE_CONFIG_PATH,
            export_matrix_path=EMBEDDING_EXPORT
        )