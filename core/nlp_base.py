from ..models.load_spacy import load_spacy_model
from .tokenizer import tokenize_text
from .parser import parse_dependencies
from .ner import extract_entities

class NLQ:
    def __init__(self):
        self.nlp = load_spacy_model()

    def analyze(self, text):
        doc = self.nlp(text)

        return {
            "tokens": tokenize_text(doc),
            "dependencies": parse_dependencies(doc),
            "entities": extract_entities(doc),
        }
