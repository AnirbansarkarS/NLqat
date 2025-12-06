from pinecone import Pinecone
from .base import VectorStoreBase

class PineconeStore(VectorStoreBase):
    def __init__(self, api_key, index_name="nlq-index"):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def add_embeddings(self, embeddings, metadatas=None):
        vectors = []
        for i, emb in enumerate(embeddings):
            vectors.append({
                "id": str(i),
                "values": emb,
                "metadata": metadatas[i] if metadatas else None
            })
        self.index.upsert(vectors=vectors)

    def query(self, embedding, top_k=3):
        res = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return res.matches
