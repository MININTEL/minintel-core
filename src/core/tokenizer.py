"""
Machine Intelligence Node - Tokenizer Module

This module provides multiple tokenization strategies, including Byte Pair Encoding (BPE) 
and WordPiece, for flexible text processing in Machine Intelligence Node models.

Author: Machine Intelligence Node Development Team
"""

import re
import json
import torch

class BaseTokenizer:
    """
    Abstract tokenizer class defining a common interface for all tokenization strategies.
    """
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def tokenize(self, text):
        """
        Convert raw text into a sequence of tokens.
        """
        raise NotImplementedError("Tokenize method must be implemented in subclasses.")

    def detokenize(self, tokens):
        """
        Convert a sequence of tokens back into readable text.
        """
        raise NotImplementedError("Detokenize method must be implemented in subclasses.")

    def encode(self, text):
        """
        Convert text into token IDs.
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get("<UNK>")) for token in tokens]

    def decode(self, token_ids):
        """
        Convert token IDs back into text.
        """
        tokens = [self.inverse_vocab.get(id, "<UNK>") for id in token_ids]
        return self.detokenize(tokens)

class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) Tokenizer for subword-based tokenization.
    """
    def __init__(self, vocab_path=None):
        super().__init__()
        if vocab_path:
            self.load_vocab(vocab_path)

    def tokenize(self, text):
        """
        Tokenizes text using BPE.
        """
        text = text.lower()
        tokens = re.findall(r"[\w]+|[^\s\w]", text)
        return tokens

    def detokenize(self, tokens):
        """
        Converts tokens back to text.
        """
        return " ".join(tokens).replace(" ##", "")

    def load_vocab(self, vocab_path):
        """
        Loads a vocabulary from a JSON file.
        """
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def save_vocab(self, vocab_path):
        """
        Saves the vocabulary to a JSON file.
        """
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f, indent=4)

class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece Tokenizer, commonly used in transformer models like BERT.
    """
    def __init__(self, vocab_path=None):
        super().__init__()
        if vocab_path:
            self.load_vocab(vocab_path)

    def tokenize(self, text):
        """
        Tokenizes text using WordPiece.
        """
        text = text.lower()
        tokens = text.split()
        output_tokens = []
        for token in tokens:
            if token in self.vocab:
                output_tokens.append(token)
            else:
                output_tokens.extend(self._split_unknown_token(token))
        return output_tokens

    def _split_unknown_token(self, token):
        """
        Splits unknown tokens into subwords.
        """
        subwords = []
        for i in range(len(token)):
            sub = token[:i+1]
            if sub in self.vocab:
                subwords.append(sub)
        if not subwords:
            subwords.append("<UNK>")
        return subwords

    def detokenize(self, tokens):
        """
        Converts tokens back to text.
        """
        return " ".join(tokens).replace(" ##", "")

# Example usage
if __name__ == "__main__":
    tokenizer = BPETokenizer()
    example_text = "Machine Intelligence Node is the future of AI."
    tokens = tokenizer.tokenize(example_text)
    token_ids = tokenizer.encode(example_text)
    decoded_text = tokenizer.decode(token_ids)

    print(f"Original Text: {example_text}")
    print(f"Tokenized: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded Text: {decoded_text}")
