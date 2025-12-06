from .base_vector_store import VectorStore
import chromadb
from typing import List, Dict, Any
import numpy as np
import uuid

class ChromaStore(VectorStore):
    def __init__(self, collection_name="hybrid_ai_docs", path="./chroma_db", persistent=True):
        if persistent:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()
            
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]] = None):
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
            
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Convert numpy embeddings to list
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        if isinstance(query_embedding_list, list) and not isinstance(query_embedding_list[0], list):
             # Ensure query is a list of lists if single query
             query_embedding_list = [query_embedding_list]

        results = self.collection.query(
            query_embeddings=query_embedding_list,
            n_results=top_k
        )
        
        output = []
        if results['documents']:
             for i, doc in enumerate(results['documents'][0]):
                 output.append({
                     "text": doc,
                     "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                     "distance": results['distances'][0][i] if results['distances'] else 0.0
                 })
        return output
