import spacy
from .tokenizer import Tokenizer
from .parser import Parser
from .ner import NER

# Lazy imports for semantic modules to avoid overhead if not used
# In a real production app, we might handle dependency injection better.

class NLP:
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
        self._summarizer = None
        self._clusterer = None

    @property
    def embedder(self):
        if self._embedder is None:
            # Import here to avoid circular dependencies or early loading
            from nlqat.semantic.embedding import Embedder
            self._embedder = Embedder()
        return self._embedder

    @property
    def summarizer_model(self):
        if self._summarizer is None:
            from nlqat.semantic.summarizer import Summarizer
            self._summarizer = Summarizer()
        return self._summarizer
    
    @property
    def clusterer_model(self):
        if self._clusterer is None:
            from nlqat.semantic.clustering import Clusterer
            self._clusterer = Clusterer()
        return self._clusterer

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

    def similarity(self, text1, text2):
        from nlqat.semantic.similarity import calculate_cosine_similarity
        vec1 = self.embed(text1)
        vec2 = self.embed(text2)
        return calculate_cosine_similarity(vec1, vec2)

    def summarize(self, text, max_length=130, min_length=30):
        return self.summarizer_model.summarize(text, max_length, min_length)
    
    def cluster(self, texts):
        embeddings = self.embed(texts)
        return self.clusterer_model.cluster(embeddings, texts)

    # --- Vector DB Integration ---

    def create_vector_store(self, store_type: str = "chroma", **kwargs):
        """
        Creates and returns a vector store instance.
        Supported types: 'chroma', 'faiss', 'pinecone'
        """
        if store_type == "chroma":
            from nlqat.vector_store.chroma_store import ChromaStore
            return ChromaStore(**kwargs)
        elif store_type == "faiss":
            from nlqat.vector_store.faiss_store import FaissStore
            return FaissStore(**kwargs)
        elif store_type == "pinecone":
            from nlqat.vector_store.pinecone_store import PineconeStore
            if PineconeStore is None:
                raise ImportError("Pinecone client not installed or failed to import.")
            return PineconeStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")


    def create_llm(self, type="openai", **kwargs):
        if type == "openai":
            from nlqat.models.openai_llm import OpenAILLM
            return OpenAILLM(**kwargs)
        elif type == "huggingface":
            from nlqat.models.huggingface_llm import HuggingFaceLLM
            return HuggingFaceLLM(**kwargs)
        else:
            raise ValueError(f"Unknown LLM type: {type}")

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
