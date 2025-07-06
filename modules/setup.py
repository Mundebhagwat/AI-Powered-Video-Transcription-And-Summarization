import nltk
import warnings
import os
from pathlib import Path

# List of required NLTK resources
REQUIRED_NLTK_RESOURCES = [
    'stopwords',       # For stopword removal
    'punkt',           # For sentence tokenization
    'wordnet',         # For lemmatization
    'averaged_perceptron_tagger',  # For POS tagging (often needed)
    'maxent_ne_chunker',  # For named entity recognition
    'words'            # For word tokenization
]

def setup_nltk_resources():
    """Ensure all required NLTK resources are available"""
    required_resources = [
        'punkt',       # Tokenizer
        'punkt_tab',   # Alternative tokenizer
        'stopwords',   # For stopword removal
        'wordnet',     # For lemmatization
        'averaged_perceptron_tagger'  # For POS tagging
    ]
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt_tab' else nltk.data.find(resource)
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

def suppress_warnings():
    """Suppress common warnings"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Suppress NLTK specific warnings if needed
    warnings.filterwarnings("ignore", module="nltk")


if __name__ == "__main__":
    setup_nltk_resources()
    suppress_warnings()
    print("NLTK resources setup complete.")