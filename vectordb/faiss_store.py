import faiss
import numpy as np
from .base import VectorStoreBase

class FaissStore(VectorStoreBase):
    def __init__(self, dim=None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim) if dim else None
        self.metadatas = []

    def add_embeddings(self, embeddings, metadatas=None):
        vectors = np.array(embeddings).astype("float32")
        
        # Lazy initialization if index is not set
        if self.index is None:
            self.dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)
            
        self.index.add(vectors)
        if metadatas:
            self.metadatas.extend(metadatas)

    def query(self, embedding, top_k=3):
        if self.index is None:
            return []
            
        vec = np.array([embedding]).astype("float32")
        scores, ids = self.index.search(vec, top_k)
        
        results = []
        for i, score in zip(ids[0], scores[0]):
            if i < 0: continue # Invalid index
            results.append({
                "metadata": self.metadatas[i] if i < len(self.metadatas) else None,
                "score": float(score)
            })
        
        return results

    def add_documents(self, texts, embeddings):
        metadatas = [{"text": t} for t in texts]
        self.add_embeddings(embeddings, metadatas=metadatas)

    def search(self, embedding, top_k=3):
        res = self.query(embedding, top_k)
        formatted = []
        for r in res:
            entry = r['metadata'].copy() if r['metadata'] else {}
            entry['distance'] = r['score']
            formatted.append(entry)
        return formatted
