from .base_vector_store import VectorStore
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import numpy as np
import uuid
import os

class PineconeStore(VectorStore):
    def __init__(self, api_key=None, index_name="hybrid-ai-index", dimension=384):
        if api_key is None:
            # Try env var
            api_key = os.environ.get("PINECONE_API_KEY")
            
        if not api_key:
             raise ValueError("Pinecone API key required.")
             
        self.pc = Pinecone(api_key=api_key)
        
        # Check if index exists, else create
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
        self.index = self.pc.Index(index_name)

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]] = None):
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        
        vectors = []
        for i, text in enumerate(texts):
            # Pinecone expects (id, vector, metadata)
            # Add text to metadata so we can retrieve it
            meta = metadatas[i].copy()
            meta["text"] = text
            
            vec = embeddings[i].tolist() if isinstance(embeddings, np.ndarray) else embeddings[i]
            
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": meta
            })
            
        # Batch upsert (simplified)
        self.index.upsert(vectors=vectors)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        if isinstance(query_vec, list) and isinstance(query_vec[0], list):
             # Flatten if needed, Pinecone query takes single vector for simple usage or we iterate
             query_vec = query_vec[0]
             
        results = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True
        )
        
        output = []
        for match in results['matches']:
            output.append({
                "text": match['metadata'].get('text', ""),
                "metadata": match['metadata'],
                "score": match['score']
            })
        return output
