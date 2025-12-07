from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedder with a SentenceTransformer model.
        Default model 'all-MiniLM-L6-v2' is small and fast.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embeddings for a string or a list of strings.
        Returns a numpy array.
        """
        return self.model.encode(text)
