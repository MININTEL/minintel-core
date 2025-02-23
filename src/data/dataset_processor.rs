//! Machine Intelligence Node - High-Performance Dataset Processor
//!
//! Provides multi-threaded dataset preprocessing, including normalization, 
//! token filtering, and augmentation, optimized for large-scale AI workflows.
//!
//! Author: Machine Intelligence Node Development Team

use rayon::prelude::*;
use regex::Regex;
use unicode_normalization::UnicodeNormalizer;
use std::collections::HashSet;

/// Struct for handling dataset preprocessing
pub struct DatasetProcessor {
    stopwords: HashSet<String>,
    remove_punctuation: bool,
    lowercase: bool,
}

impl DatasetProcessor {
    /// Initializes a new dataset processor instance
    ///
    /// # Arguments
    /// * `stopwords` - A set of stopwords to filter out.
    /// * `remove_punctuation` - Whether to remove punctuation from text.
    /// * `lowercase` - Whether to convert text to lowercase.
    pub fn new(stopwords: HashSet<String>, remove_punctuation: bool, lowercase: bool) -> Self {
        DatasetProcessor {
            stopwords,
            remove_punctuation,
            lowercase,
        }
    }

    /// Applies full preprocessing pipeline to a dataset
    ///
    /// # Arguments
    /// * `dataset` - Vector of strings representing text samples.
    ///
    /// # Returns
    /// * Processed dataset as a vector of cleaned strings.
    pub fn preprocess_dataset(&self, dataset: Vec<String>) -> Vec<String> {
        dataset
            .par_iter()
            .map(|text| self.preprocess_text(text))
            .collect()
    }

    /// Applies preprocessing transformations to a single text sample
    ///
    /// # Arguments
    /// * `text` - A single text input.
    ///
    /// # Returns
    /// * Cleaned and preprocessed text.
    fn preprocess_text(&self, text: &str) -> String {
        let mut processed_text = text.to_string();

        // Normalize Unicode characters
        processed_text = processed_text.nfkc().collect();

        // Convert to lowercase if enabled
        if self.lowercase {
            processed_text = processed_text.to_lowercase();
        }

        // Remove punctuation if enabled
        if self.remove_punctuation {
            let re = Regex::new(r"[^\w\s]").unwrap();
            processed_text = re.replace_all(&processed_text, "").to_string();
        }

        // Tokenize and remove stopwords
        let words: Vec<&str> = processed_text.split_whitespace().collect();
        let filtered_words: Vec<&str> = words
            .iter()
            .filter(|&&word| !self.stopwords.contains(word))
            .cloned()
            .collect();

        filtered_words.join(" ")
    }
}

/// Example Usage
fn main() {
    use std::iter::FromIterator;
    
    let stopwords = HashSet::from_iter(vec!["the".to_string(), "is".to_string(), "and".to_string()]);
    let processor = DatasetProcessor::new(stopwords, true, true);

    let dataset = vec![
        "Machine Intelligence Node is optimizing AI!".to_string(),
        "Rust-powered text processing is super fast.".to_string(),
    ];

    let processed_data = processor.preprocess_dataset(dataset);
    
    for text in processed_data {
        println!("{}", text);
    }
}
