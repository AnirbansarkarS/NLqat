class VectorStoreBase:
    def add_embeddings(self, embeddings, metadatas=None):
        raise NotImplementedError
    
    def query(self, embedding, top_k=3):
        raise NotImplementedError
