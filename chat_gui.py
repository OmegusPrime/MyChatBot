import os
import sys
import tkinter as tk
from tkinter import scrolledtext
import threading
import numpy as np
import mysql.connector

# Resolve workspace root so sibling modules import smoothly
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Custom Model Layer and Component Imports ---
from Model.Transformer import TransformerCoreStack
from Model.Layers import LayerNormScratch, SinusoidalPositionalEncoding, PositionWiseFeedForward
from Model.Attention import CausalMultiHeadAttention
from Model.Generator import LanguageModelHead, ChatBotInferenceEngine
from Embedding.DenseEmbeddingEngine import DenseEmbeddingEngine

# --- MySQL Configuration Database Credentials ---
MYSQL_DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Dhruv356564@@##',  # Replace with your recovered MySQL root password
    'database': 'chatbot_db'
}


class DatabaseBackedTokenizer:
    def __init__(self, mysql_config):
        self.config = mysql_config
        self.vocab = {}
        self.inv_vocab = {}
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

        cursor.execute("SELECT COUNT(*) FROM chatbot_vocabulary")
        if cursor.fetchone()[0] == 0:
            base_tokens = [(0, '<pad>'), (1, '<unk>'), (2, '<bos>'), (3, '<eos>')]
            cursor.executemany("INSERT INTO chatbot_vocabulary (tokenId, token) VALUES (%s, %s)", base_tokens)
            conn.commit()

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
        words = text_string.strip().split()
        return [self.get_or_create_token_id(w) for w in words]

    def decode(self, id_list):
        return " ".join([self.inv_vocab.get(t_id, "<unk>") for t_id in id_list])


class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MySQL Dynamic AI Chat Interface")
        self.root.geometry("450x600")
        self.root.configure(bg="#1e1e1e")

        # FIX 1: Initialize a persistent running memory history list for full conversations
        self.conversation_history = []

        print("Connecting to local running MySQL Instance...")
        self.tokenizer = DatabaseBackedTokenizer(MYSQL_DB_CONFIG)
        self.initialize_chatbot_backend()
        self.setup_ui_layout()

    def initialize_chatbot_backend(self):
        vocab_size = max(len(self.tokenizer.vocab), 10)

        self.embed_engine = DenseEmbeddingEngine(vocab_size=vocab_size, embedding_dim=32)
        self.transformer = TransformerCoreStack(vocab_size=vocab_size, embed_dim=32, num_blocks=2)
        self.language_head = LanguageModelHead(embed_dim=32, vocab_size=vocab_size)

        self.engine = ChatBotInferenceEngine(
            transformer_stack=self.transformer,
            lm_head=self.language_head,
            embedding_lookup_func=self.embed_engine.get_token_vector
        )

    def setup_ui_layout(self):
        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, bg="#2d2d2d", fg="#ffffff", font=("Arial", 11), state=tk.DISABLED
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(self.root, bg="#1e1e1e")
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.entry_field = tk.Entry(
            input_frame, bg="#3d3d3d", fg="#ffffff", insertbackground="white", font=("Arial", 11), relief=tk.FLAT
        )
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.entry_field.bind("<Return>", self.on_send_triggered)

        send_button = tk.Button(
            input_frame, text="Send", bg="#007acc", fg="#ffffff", font=("Arial", 10, "bold"),
            relief=tk.FLAT, command=self.on_send_triggered
        )
        send_button.pack(side=tk.RIGHT, padx=(5, 0), ipady=5, ipadx=15)
        self.append_message_to_window("System", "MySQL Connected. Tokens register automatically on input turns!")

    def append_message_to_window(self, sender, text):
        self.chat_display.config(state=tk.NORMAL)
        if sender == "You":
            self.chat_display.insert(tk.END, f"\nYou: {text}\n", "user_tag")
        elif sender == "System":
            self.chat_display.insert(tk.END, f"[{text}]\n", "system_tag")
        else:
            self.chat_display.insert(tk.END, f"\nChatbot: {text}\n", "bot_tag")

        self.chat_display.tag_config("user_tag", foreground="#4fc1ff", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("bot_tag", foreground="#9cdcfe")
        self.chat_display.tag_config("system_tag", foreground="#6a9955", font=("Arial", 9, "italic"))
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def on_send_triggered(self, event=None):
        user_text = self.entry_field.get().strip()
        if not user_text:
            return

        self.entry_field.delete(0, tk.END)
        self.append_message_to_window("You", user_text)

        worker_thread = threading.Thread(target=self.process_bot_response, args=(user_text,))
        worker_thread.start()

    def process_bot_response(self, prompt_text):
        try:
            # 1. Map user query strings to Token IDs (and expand MySQL)
            new_input_ids = self.tokenizer.encode_dynamic_query(prompt_text)

            # FIX 2: Append the new user input tokens to the running history array
            self.conversation_history.extend(new_input_ids)

            # 2. Check and scale matrix dimensions to handle vocabulary growth
            current_vocab_count = len(self.tokenizer.vocab)
            self.embed_engine.expand_embedding_matrix(current_vocab_count)
            self.language_head.expand_projection_head(current_vocab_count)

            # Update inference engine configuration references
            self.engine.vocab_size = current_vocab_count

            # Slidely bound conversation context to prevent exceeding maximum matrix length (512 tokens)
            if len(self.conversation_history) > 400:
                self.conversation_history = self.conversation_history[-400:]

            # FIX 3: Pass the cumulative conversation history into the generation manager
            output_sequence_ids = self.engine.(
                initial_token_ids=self.conversation_history,
                max_nex_tokens=10,
                tokenizer_eos_id=3,
                top_k=10,
                top_p=0.8
            )

            # Extract ONLY the newly generated tokens appended at the end of the input sequence
            newly_generated_ids = output_sequence_ids[len(self.conversation_history):]

            # FIX 4: Append the chatbot's own generated response tokens to the ongoing history context
            self.conversation_history.extend(newly_generated_ids)

            # Decode numbers back to whitespace-separated strings
            response_string = self.tokenizer.decode(newly_generated_ids)

            if not response_string.strip():
                response_string = "..."

            self.root.after(0, self.append_message_to_window, "Chatbot", response_string)

        except Exception as err:
            self.root.after(0, self.append_message_to_window, "System", f"Runtime Error: {str(err)}")


if __name__ == "__main__":
    root_window = tk.Tk()
    app = ChatBotGUI(root_window)
    root_window.mainloop()