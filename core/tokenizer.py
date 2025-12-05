import spacy
from typing import List

class Tokenizer:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [token.text for token in doc]

    def lemmatize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def pos_tag(self, text: str) -> List[tuple]:
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
