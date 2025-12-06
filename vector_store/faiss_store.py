from .base_vector_store import VectorStore
import faiss
import numpy as np
from typing import List, Dict, Any

class FaissStore(VectorStore):
    def __init__(self, dimension: int = 384):
        # Default dimension for all-MiniLM-L6-v2 is 384
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []  # FAISS doesn't store text, so we keep it here
        self.metadatas = []

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]] = None):
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
            
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        
        if isinstance(embeddings, np.ndarray):
            # FAISS expects float32
            self.index.add(embeddings.astype('float32'))
        else:
            self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if isinstance(query_embedding, np.ndarray):
             query_vec = query_embedding.astype('float32')
        else:
             query_vec = np.array(query_embedding).astype('float32')
             
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
            
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(distances[0][i])
                })
        return results
