from .base_pipeline import NLP
from typing import List, Dict, Any, Optional

class Pipeline:
    def __init__(
        self,
        enable_spacy: bool = True,
        vector_store_type: Optional[str] = "chroma",
        vector_store_path: str = "./chroma_db",
        llm = None
    ):
        """
        Initializes the unified pipeline.
        
        Args:
            enable_spacy: Whether to enable classic NLP (POS, NER, etc.)
            vector_store_type: 'chroma', 'faiss', 'pinecone', or None.
            vector_store_path: Path for local vector stores.
            llm: An instance of an LLM class (e.g. OpenAI_LLM).
        """
        self.nlp = NLP() if enable_spacy else None
        self.vector_store = None
        self.llm = llm

        if vector_store_type is not None:
             # Ensure NLP is initialized for embedding
             if self.nlp is None:
                 self.nlp = NLP()
             self.vector_store = self.nlp.create_vector_store(vector_store_type, path=vector_store_path)

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Adds documents to the pipeline's vector store.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not enabled.")
        
        print("Generating embeddings...")
        embeddings = self.nlp.embed(texts)
        
        print(f"Adding {len(texts)} documents to {self.vector_store.__class__.__name__}...")
        self.vector_store.add_documents(texts, embeddings, metadatas)

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Analyze question (optional)
        2. Retrieve relevant docs from VectorStore
        3. Send to LLM with context
        """
        result = {
            "question": question,
            "analysis": None,
            "retrieved_docs": [],
            "answer": None
        }

        # 1. Linguistic Analysis
        if self.nlp:
            doc = self.nlp.analyze(question)
            result["analysis"] = {
                "tokens": doc.tokens,
                "entities": doc.entities,
                "pos_tags": doc.pos_tags
            }

        # 2. Retrieval
        context = ""
        if self.vector_store:
            print(f"Retrieving docs for: '{question}'...")
            q_emb = self.nlp.embed(question)
            retrieved = self.vector_store.search(q_emb, top_k=top_k)
            result["retrieved_docs"] = retrieved
            
            context = "\n\n".join([doc['text'] for doc in retrieved])

        # 3. LLM Generation
        if self.llm:
            if context:
                prompt_content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based on the context above."
            else:
                prompt_content = question
                
            print("Querying LLM...")
            # Assuming LLM has a generate method or similar. 
            # Adapting to common interface if needed.
            if hasattr(self.llm, "generate_response"):
                 answer = self.llm.generate_response(prompt_content)
            elif hasattr(self.llm, "generate"):
                 answer = self.llm.generate(prompt_content)
            else:
                 answer = str(self.llm(prompt_content)) # Try callable
            
            result["answer"] = answer
        else:
            result["answer"] = "LLM not configured."

        return result
