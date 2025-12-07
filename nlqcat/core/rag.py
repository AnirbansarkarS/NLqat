from typing import List
from .base_pipeline import NLP, Document

class RAGPipeline:
    def __init__(self, nlp: NLP, vector_store, llm):
        """
        Initialize RAG Pipeline.
        nlp: The NLP pipeline instance.
        vector_store: An initialized VectorStore (Chroma, etc).
        llm: An initialized LLM (OpenAI, HuggingFace, etc).
        """
        self.nlp = nlp
        self.vector_store = vector_store
        self.llm = llm

    def index_documents(self, documents: List[str]):
        """
        Embeds and adds documents to the vector store.
        """
        print(f"Indexing {len(documents)} documents...")
        embeddings = self.nlp.embed(documents)
        self.vector_store.add_documents(documents, embeddings)

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Retrieves context and generates an answer.
        """
        # 1. Retrieve relevant context
        q_emb = self.nlp.embed(question)
        results = self.vector_store.search(q_emb, top_k=top_k)
        
        context_texts = [res['text'] for res in results]
        context_block = "\n".join(context_texts)
        
        # 2. Construct Prompt
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        
        # 3. Generate Answer
        print("Generating answer with LLM...")
        answer = self.llm.generate(prompt)
        return answer
