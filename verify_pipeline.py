from nlqat.core.pipeline import Pipeline
from nlqat.models.mock_llm import MockLLM
import shutil
import os

def test_pipeline():
    print("Initialize Unified Pipeline...")
    
    # Use a temporary path via generic 'tmp' or just local unique dir
    test_db_path = "./pipeline_test_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        
    pipe = Pipeline(
        enable_spacy=True,
        vector_store_type="chroma",
        vector_store_path=test_db_path,
        llm=MockLLM()
    )
    
    docs = [
        "The project is named HybridAI.",
        "It combines spaCy and semantic transformers.",
        "Phase 5 is the unified pipeline engine."
    ]
    
    print("\n--- Adding Documents ---")
    pipe.add_documents(docs)
    
    print("\n--- Querying Pipeline ---")
    question = "What is the project name and what does it combine?"
    result = pipe.query(question)
    
    print("\n[Result Analysis]")
    print(f"Entities detected: {result['analysis']['entities']}")
    
    print("\n[Retrieved Context]")
    for doc in result['retrieved_docs']:
        print(f"- {doc['text']}")
        
    print("\n[LLM Answer]")
    print(result['answer'])
    
    assert "HybridAI" in str(result['retrieved_docs'])
    print("\nSUCCESS: Pipeline verified.")

if __name__ == "__main__":
    test_pipeline()
