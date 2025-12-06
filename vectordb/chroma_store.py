import chromadb
from .base import VectorStoreBase

class ChromaStore(VectorStoreBase):
    def __init__(self, path="./chroma_db"):
        self.client = chromadb.PersistentClient(path)
        self.collection = self.client.get_or_create_collection(
            name="nlq_store",
            metadata={"hnsw:space": "cosine"}
        )

    def add_embeddings(self, embeddings, metadatas=None):
        ids = [str(i) for i in range(len(embeddings))]
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, embedding, top_k=3):
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results

    def add_documents(self, texts, embeddings):
        metadatas = [{"text": t} for t in texts]
        self.add_embeddings(embeddings, metadatas=metadatas)

    def search(self, embedding, top_k=3):
        res = self.query(embedding, top_k)
        formatted = []
        # Check if we have results
        if res and res.get('metadatas') and len(res['metadatas']) > 0:
            metas = res['metadatas'][0]
            dists = res['distances'][0] if res.get('distances') else [0] * len(metas)
            
            for meta, dist in zip(metas, dists):
                entry = meta.copy()
                entry['distance'] = dist
                formatted.append(entry)
        return formatted
