"""
Machine Intelligence Node - Text Preprocessor

This module provides a robust text preprocessing pipeline, handling 
normalization, tokenization, punctuation removal, and stopword filtering.

Author: Machine Intelligence Node Development Team
"""

import re
import string
import unicodedata
from typing import List, Optional
from nltk.corpus import stopwords
from transformers import AutoTokenizer

class TextPreprocessor:
    """
    A highly flexible text preprocessor supporting multiple cleaning techniques.
    """
    def __init__(
        self,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        language: str = "english",
        custom_token_filter: Optional[List[str]] = None,
        tokenizer_model: Optional[str] = None,
    ):
        """
        Initializes the text preprocessor.

        Args:
            remove_punctuation (bool): Whether to strip punctuation from text.
            lowercase (bool): Whether to convert text to lowercase.
            remove_stopwords (bool): Whether to remove stopwords.
            language (str): Language for stopword filtering (default: English).
            custom_token_filter (List[str], optional): List of custom tokens to remove.
            tokenizer_model (str, optional): Pretrained tokenizer model (Hugging Face).
        """
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.language = language
        self.custom_token_filter = set(custom_token_filter) if custom_token_filter else set()
        self.stopwords = set(stopwords.words(language)) if remove_stopwords else set()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model) if tokenizer_model else None

    def normalize_text(self, text: str) -> str:
        """
        Normalizes text by converting Unicode characters to NFKC form.
        
        Args:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        return unicodedata.normalize("NFKC", text)

    def remove_punctuation_marks(self, text: str) -> str:
        """
        Removes punctuation from text while preserving key symbols.

        Args:
            text (str): Input text.

        Returns:
            str: Cleaned text without punctuation.
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_custom_tokens(self, tokens: List[str]) -> List[str]:
        """
        Removes predefined unwanted tokens from a list.

        Args:
            tokens (List[str]): List of tokenized words.

        Returns:
            List[str]: Filtered token list.
        """
        return [token for token in tokens if token not in self.custom_token_filter]

    def preprocess(self, text: str) -> List[str]:
        """
        Applies the full preprocessing pipeline.

        Args:
            text (str): Raw text input.

        Returns:
            List[str]: Cleaned and tokenized text.
        """
        if self.lowercase:
            text = text.lower()
        
        text = self.normalize_text(text)

        if self.remove_punctuation:
            text = self.remove_punctuation_marks(text)

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        if self.custom_token_filter:
            tokens = self.remove_custom_tokens(tokens)

        return tokens

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes text using a pre-trained tokenizer.

        Args:
            text (str): Input text.

        Returns:
            List[int]: Tokenized output in ID format.
        """
        if not self.tokenizer:
            raise ValueError("No tokenizer model specified.")
        
        return self.tokenizer(text, padding="max_length", truncation=True)["input_ids"]

# Example usage
if __name__ == "__main__":
    sample_text = "Machine Intelligence Node is revolutionizing AI!"

    # Basic Preprocessor
    preprocessor = TextPreprocessor(remove_punctuation=True, lowercase=True, remove_stopwords=True)
    processed_text = preprocessor.preprocess(sample_text)
    print(f"Processed Text: {processed_text}")

    # Tokenizer-based Preprocessor
    tokenizer_preprocessor = TextPreprocessor(tokenizer_model="bert-base-uncased")
    tokenized_output = tokenizer_preprocessor.tokenize(sample_text)
    print(f"Tokenized Output: {tokenized_output}")
