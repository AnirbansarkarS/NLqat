from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]] = None):
        """
        Adds documents to the store.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the most similar documents.
        """
        pass
