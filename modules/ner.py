import spacy
import subprocess

def perform_ner(transcription):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(transcription)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    unique_entities = list(set(entities))

    return unique_entities
