import spacy
from typing import List, Optional

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

class Parser:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def parse(self, text: str) -> list:
        doc = self.nlp(text)
        parsed_data = []
        for token in doc:
            parsed_data.append({
                "text": token.text,
                "dep": token.dep_,
                "head": token.head.text,
                "head_pos": token.head.pos_,
                "children": [child.text for child in token.children]
            })
        return parsed_data

class NER:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def extract_entities(self, text: str) -> list:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities

class Document:
    def __init__(self, text, tokens, lemmas, pos_tags, entities, dependencies, spacy_doc):
        self.text = text
        self.tokens = tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.entities = entities
        self.dependencies = dependencies
        self.doc = spacy_doc

    def __repr__(self):
        return f"<Document with {len(self.tokens)} tokens, {len(self.entities)} entities>"

class SpacyEngine:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
        
        self.tokenizer = Tokenizer(self.nlp)
        self.parser = Parser(self.nlp)
        self.ner = NER(self.nlp)
        
        # Semantic modules (initialized lazily)
        self._embedder = None
        # self._summarizer = None
        # self._clusterer = None

    @property
    def embedder(self):
        if self._embedder is None:
            # Import here to avoid circular dependencies or early loading
            from ..embeddings.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def analyze(self, text: str):
        """
        Runs the full NLP pipeline and returns a structured object.
        """
        doc = self.nlp(text)
        
        return Document(
            text=text,
            tokens=self.tokenizer.tokenize(text),
            lemmas=self.tokenizer.lemmatize(text),
            pos_tags=self.tokenizer.pos_tag(text),
            entities=self.ner.extract_entities(text),
            dependencies=self.parser.parse(text),
            spacy_doc=doc
        )

    # --- Semantic Features ---

    def embed(self, text):
        return self.embedder.embed(text)

    # --- Vector DB Integration ---

    def create_vector_store(self, store_type: str = "chroma", **kwargs):
        """
        Creates and returns a vector store instance.
        """
        if store_type == "chroma":
            from ..vectorstore.chroma_client import ChromaClient
            return ChromaClient(**kwargs)
        else:
            raise ValueError(f"Unknown or unsupported vector store type: {store_type}")

    # --- LLM Integration ---
    # Moved to explicit instantiation in pipeline or usage code.
