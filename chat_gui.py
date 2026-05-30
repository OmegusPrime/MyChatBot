"""
Filename: chat_gui.py
Description: A native desktop GUI interface for the scratch chatbot engine
             built using Tkinter and integrated with your custom model pipeline.
"""

import os
import sys
import tkinter as tk
from tkinter import scrolledtext
import threading

# Resolve workspace root (D:\MyChatBot) so modules import smoothly
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Import your custom system components ---
from Pipeline.tokenizer import Tokenizer
from Embedding.DenseEmbeddingEngine import DenseEmbeddingEngine
from Model.Transformer import TransformerCoreStack
from Model.Generator import LanguageModelHead, ChatBotInferenceEngine


class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scratch AI Chatbot Interface")
        self.root.geometry("450x600")
        self.root.configure(bg="#1e1e1e")

        print("Loading backend neural components for GUI...")
        self.initialize_chatbot_backend()
        self.setup_ui_layout()

    def initialize_chatbot_backend(self):
        """Instantiates and links your scratch pipeline parts together."""
        BPE_CONFIG_PATH = os.path.join(BASE_DIR, "Pipeline", "bpe.json")
        EMBEDDING_EXPORT = os.path.join(BASE_DIR, "Embedding", "trained_chatbot_embeddings.npz")

        # 1. Load Tokenizer
        if os.path.exists(BPE_CONFIG_PATH):
            self.tokenizer = Tokenizer.from_pretrained(BPE_CONFIG_PATH)
        else:
            raise FileNotFoundError(f"Missing bpe.json at {BPE_CONFIG_PATH}. Please run main_and_eval.py first.")

        vocab_size = self.tokenizer.vocab_size

        # 2. Load Word Vectors
        self.embed_engine = DenseEmbeddingEngine(vocab_size=vocab_size, embedding_dim=32)
        if os.path.exists(EMBEDDING_EXPORT):
            self.embed_engine.load_weights(EMBEDDING_EXPORT)
        else:
            print("Warning: Pre-trained matrix weights not found. Running on random weights.")

        # 3. Assemble Core Transformer Block Stack & Language Head
        self.transformer = TransformerCoreStack(vocab_size=vocab_size, embed_dim=32, num_blocks=2)
        self.language_head = LanguageModelHead(embed_dim=32, vocab_size=vocab_size)

        # 4. Bind into the Inference Engine
        self.engine = ChatBotInferenceEngine(
            transformer_stack=self.transformer,
            lm_head=self.language_head,
            embedding_lookup_func=self.embed_engine.get_token_vector
        )
        print("Backend loaded successfully.")

    def setup_ui_layout(self):
        """Constructs widgets for the graphical application window."""
        # Chat History Display Box
        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, bg="#2d2d2d", fg="#ffffff",
            font=("Arial", 11), state=tk.DISABLED
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Bottom Input Container Frame
        input_frame = tk.Frame(self.root, bg="#1e1e1e")
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        # User Text Entry Field
        self.entry_field = tk.Entry(
            input_frame, bg="#3d3d3d", fg="#ffffff", insertbackground="white",
            font=("Arial", 11), relief=tk.FLAT
        )
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.entry_field.bind("<Return>", self.on_send_triggered)  # Send on Enter keypress

        # Send Button
        send_button = tk.Button(
            input_frame, text="Send", bg="#007acc", fg="#ffffff",
            activebackground="#005999", activeforeground="#ffffff",
            font=("Arial", 10, "bold"), relief=tk.FLAT, command=self.on_send_triggered
        )
        send_button.pack(side=tk.RIGHT, padx=(5, 0), ipady=5, ipadx=15)

        # Display welcoming system message
        self.append_message_to_window("System", "Chatbot online. Type a message to begin!")

    def append_message_to_window(self, sender, text):
        """Safely injects text segments into the disabled view widget."""
        self.chat_display.config(state=tk.NORMAL)
        if sender == "You":
            self.chat_display.insert(tk.END, f"\nYou: {text}\n", "user_tag")
        elif sender == "System":
            self.chat_display.insert(tk.END, f"[{text}]\n", "system_tag")
        else:
            self.chat_display.insert(tk.END, f"\nChatbot: {text}\n", "bot_tag")

        # Style formatting tags
        self.chat_display.tag_config("user_tag", foreground="#4fc1ff", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("bot_tag", foreground="#9cdcfe")
        self.chat_display.tag_config("system_tag", foreground="#6a9955", font=("Arial", 9, "italic"))

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)  # Autoscroll to bottom

    def on_send_triggered(self, event=None):
        """Extracts input text and hands processing over to a background thread."""
        user_text = self.entry_field.get().strip()
        if not user_text:
            return

        self.entry_field.delete(0, tk.END)  # Clear text area field
        self.append_message_to_window("You", user_text)

        # Use threading so the graphical window window doesn't crash during matrix loops
        worker_thread = threading.Thread(target=self.process_bot_response, args=(user_text,))
        worker_thread.start()

    def process_bot_response(self, prompt_text):
        """Runs custom BPE encodings, model transformations, and structural decoding."""
        try:
            # Encode incoming prompt via custom BPE parameters
            input_ids = self.tokenizer.encode(prompt_text, add_special=False)

            # Fallback if text contains entirely unmapped zero tokens
            if not input_ids:
                self.root.after(0, self.append_message_to_window, "Chatbot", "...")
                return

            # Execute autoregressive prediction loop via your local core blocks
            output_sequence_ids = self.engine.generate_response(
                initial_token_ids=input_ids,
                max_new_tokens=15,
                tokenizer_eos_id=3,
                top_k=20,
                top_p=0.85
            )

            # Strip input tokens from the array sequence to isolate the newly generated string
            generated_only_ids = output_sequence_ids[len(input_ids):]

            # Decode numbers back to whitespace-preserved unicode strings
            response_string = self.tokenizer.decode(generated_only_ids, skip_special=True)

            if not response_string.strip():
                response_string = "(The model produced an empty token string.)"

            # Thread-safe UI update call using Tkinter's 'after' method
            self.root.after(0, self.append_message_to_window, "Chatbot", response_string)

        except Exception as err:
            self.root.after(0, self.append_message_to_window, "System", f"Inference Error: {str(err)}")


if __name__ == "__main__":
    root_window = tk.Tk()
    app = ChatBotGUI(root_window)
    root_window.mainloop()