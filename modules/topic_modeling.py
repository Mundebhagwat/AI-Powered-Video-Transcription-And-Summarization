from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
import nltk
import re
import os

def preprocess_text(transcription):
    stop_words = set(stopwords.words('english')).union(STOPWORDS)
    tokens = word_tokenize(transcription.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]

    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def preprocess_and_split(transcription):
    sentences = re.split(r'(?<=[.!?])\s+', transcription)
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    processed_sentences = [sentence for sentence in processed_sentences if sentence]
    return processed_sentences

def build_lda_model(text_data, num_topics=5):
    if not text_data:
        raise ValueError("The corpus is empty after preprocessing. Please check the input data.")

    dictionary = corpora.Dictionary(text_data)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    if not corpus:
        raise ValueError("Cannot compute LDA over an empty collection (no terms).")

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=50,
                         alpha='auto', eta='auto', random_state=42)
    return lda_model

def print_and_save_topics(lda_model, num_words=3, output_file="topics.txt", output_dir="data/output"):
    topics = lda_model.print_topics(num_words=num_words)

    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    with open(os.path.join(output_dir, output_file), 'w') as file:
        for topic_id, topic in topics:
            topic_words = [word.split('*"')[1].strip('"') for word in topic.split(' + ')]
            topic_str = f"Topic {topic_id}: {', '.join(topic_words)}\n"
            file.write(topic_str)
