from core.base_pipeline import NLP
from core.rag import RAGPipeline
from models.llm_base import LLMBase
import os

# Mock LLM for testing without keys/heavy models
class MockLLM(LLMBase):
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        return f"Mock Answer based on prompt: {prompt[:30]}..."

    def chat(self, messages, **kwargs) -> str:
        return "Mock Chat Response"

def test_phase4():
    print("Initializing NLP (Phase 4)...")
    nlp = NLP()
    
    # Use in-memory FAISS for speed
    print("Creating Vector Store...")
    vs = nlp.create_vector_store("faiss", dim=384) 
    
    # Create Mock LLM
    print("Creating Mock LLM...")
    llm = MockLLM()
    
    # Initialize RAG Pipeline
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline(nlp, vs, llm)
    
    # 1. Index Documents
    docs = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome."
    ]
    rag.index_documents(docs)
    
    # 2. Ask Question
    question = "What is the capital of France?"
    print(f"\nQuestion: {question}")
    
    answer = rag.answer(question)
    print(f"Answer: {answer}")
    
    # 3. Test Factory Methods (Verification of loading, not running heavy models)
    print("\nTesting NLP.create_llm factory...")
    try:
        # Just check imports work
        from models.openai_llm import OpenAILLM
        print("OpenAILLM class found.")
    except ImportError:
        print("OpenAILLM Import Failed.")

    try:
        from models.huggingface_llm import HuggingFaceLLM
        print("HuggingFaceLLM class found.")
    except ImportError:
        print("HuggingFaceLLM Import Failed.")

    print("\nPhase 4 Verification Complete.")

if __name__ == "__main__":
    test_phase4()
