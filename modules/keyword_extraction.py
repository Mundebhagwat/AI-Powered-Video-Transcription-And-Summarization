from keybert import KeyBERT
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Global KeyBERT model
kw_model = KeyBERT()

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_text

def extract_keywords(transcription, num_keywords=10):
    lemmatized_transcription = lemmatize_text(transcription)
    keywords = kw_model.extract_keywords(
        lemmatized_transcription,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=num_keywords
    )
    keywords = [kw for kw in keywords if kw[1] > 0.3]  
    return keywords
