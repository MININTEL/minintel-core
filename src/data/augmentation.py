"""
Machine Intelligence Node - Data Augmentation

Implements advanced text augmentation techniques to enhance model robustness 
by generating diverse variations of training data.

Author: Machine Intelligence Node Development Team
"""

import random
import re
import nltk
import torch
from typing import List, Optional
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer

class TextAugmenter:
    """
    A flexible text augmentation module supporting synonym replacement, 
    random insertion, deletion, and back-translation.
    """
    def __init__(
        self, 
        synonym_replacement: bool = True,
        random_insertion: bool = True,
        random_deletion: bool = True,
        back_translation: bool = False,
        back_translation_lang: str = "fr",
        character_swap: bool = False,
        noise_injection: bool = False,
    ):
        """
        Initializes the text augmentation pipeline.

        Args:
            synonym_replacement (bool): Enable synonym-based augmentation.
            random_insertion (bool): Insert random words into text.
            random_deletion (bool): Randomly delete words from text.
            back_translation (bool): Translate text to another language and back.
            back_translation_lang (str): Target language for back-translation.
            character_swap (bool): Randomly swap characters within words.
            noise_injection (bool): Inject random typos or noise into text.
        """
        self.synonym_replacement = synonym_replacement
        self.random_insertion = random_insertion
        self.random_deletion = random_deletion
        self.back_translation = back_translation
        self.back_translation_lang = back_translation_lang
        self.character_swap = character_swap
        self.noise_injection = noise_injection

        # Load back-translation model if needed
        if self.back_translation:
            self.translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{self.back_translation_lang}")
            self.tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{self.back_translation_lang}")

    def get_synonym(self, word: str) -> str:
        """
        Retrieves a synonym for a given word using WordNet.

        Args:
            word (str): Input word.

        Returns:
            str: A synonym of the input word.
        """
        synonyms = wordnet.synsets(word)
        if not synonyms:
            return word
        return synonyms[0].lemmas()[0].name()

    def synonym_replace(self, text: str) -> str:
        """
        Replaces words with their synonyms.

        Args:
            text (str): Input text.

        Returns:
            str: Text with randomly replaced synonyms.
        """
        words = text.split()
        new_words = [self.get_synonym(word) if random.random() < 0.2 else word for word in words]
        return " ".join(new_words)

    def random_insert(self, text: str) -> str:
        """
        Inserts random words into the text.

        Args:
            text (str): Input text.

        Returns:
            str: Augmented text with random insertions.
        """
        words = text.split()
        insertions = max(1, len(words) // 10)
        for _ in range(insertions):
            rand_index = random.randint(0, len(words) - 1)
            words.insert(rand_index, self.get_synonym(words[rand_index]))
        return " ".join(words)

    def random_delete(self, text: str) -> str:
        """
        Randomly deletes words from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Augmented text with words randomly removed.
        """
        words = text.split()
        new_words = [word for word in words if random.random() > 0.2]
        return " ".join(new_words) if new_words else words[0]

    def swap_characters(self, text: str) -> str:
        """
        Randomly swaps characters within words.

        Args:
            text (str): Input text.

        Returns:
            str: Augmented text with character swaps.
        """
        words = list(text)
        for _ in range(len(words) // 5):
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
        return "".join(words)

    def inject_noise(self, text: str) -> str:
        """
        Injects random typos into the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with randomly added noise.
        """
        noise_chars = ["@", "#", "$", "%", "&", "*"]
        words = list(text)
        for _ in range(len(words) // 10):
            i = random.randint(0, len(words) - 1)
            words[i] = random.choice(noise_chars)
        return "".join(words)

    def back_translate(self, text: str) -> str:
        """
        Performs back-translation (English -> Other Lang -> English).

        Args:
            text (str): Input text.

        Returns:
            str: Back-translated text.
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.translator.generate(**encoded_text)
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

        # Translate back to original language
        encoded_translated = self.tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True)
        back_translated = self.translator.generate(**encoded_translated)
        return self.tokenizer.batch_decode(back_translated, skip_special_tokens=True)[0]

    def augment(self, text: str) -> str:
        """
        Applies selected augmentations to the input text.

        Args:
            text (str): Raw text input.

        Returns:
            str: Augmented text.
        """
        if self.synonym_replacement:
            text = self.synonym_replace(text)

        if self.random_insertion:
            text = self.random_insert(text)

        if self.random_deletion:
            text = self.random_delete(text)

        if self.character_swap:
            text = self.swap_characters(text)

        if self.noise_injection:
            text = self.inject_noise(text)

        if self.back_translation:
            text = self.back_translate(text)

        return text

# Example Usage
if __name__ == "__main__":
    augmenter = TextAugmenter(synonym_replacement=True, random_insertion=True, back_translation=False)
    sample_text = "Machine Intelligence Node is optimizing AI models."
    augmented_text = augmenter.augment(sample_text)
    print(f"Original: {sample_text}")
    print(f"Augmented: {augmented_text}")
