from core.base_pipeline import NLP
import shutil
import os

def test_phase3():
    print("Initializing NLP (Phase 3 Verification)...")
    nlp = NLP()
    
    # 1. Test ChromaDB
    print("\n--- Testing ChromaDB Integration ---")
    chroma_path = "./test_chroma_db"
    
    # Clean up previous test run
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        
    vs_chroma = nlp.create_vector_store("chroma", path=chroma_path)
    
    texts = [
        "Apple released the new iPhone.",
        "Google released the Pixel phone.",
        "The sun is a star.",
        "Planets orbit the sun."
    ]
    
    print("Embedding and adding documents...")
    embeddings = nlp.embed(texts)
    vs_chroma.add_documents(texts, embeddings)
    
    query = "New smartphone release"
    print(f"Searching for: '{query}'")
    q_emb = nlp.embed(query)
    results = vs_chroma.search(q_emb, top_k=2)
    
    print("Results:")
    found_tech = False
    for res in results:
        print(f" - {res['text']} (Dist: {res.get('distance', 0):.4f})")
        if "iPhone" in res['text'] or "Pixel" in res['text']:
            found_tech = True
            
    if found_tech:
        print("SUCCESS: ChromaDB retrieved relevant documents.")
    else:
        print("FAILURE: ChromaDB did not retrieve expected documents.")

    # 2. Test FAISS
    print("\n--- Testing FAISS Integration ---")
    vs_faiss = nlp.create_vector_store("faiss") # In-memory
    vs_faiss.add_documents(texts, embeddings)
    
    print(f"Searching FAISS for: '{query}'")
    results_faiss = vs_faiss.search(q_emb, top_k=2)
    
    print("Results:")
    found_tech_faiss = False
    for res in results_faiss:
        print(f" - {res['text']} (Dist: {res.get('distance', 0):.4f})")
        if "iPhone" in res['text'] or "Pixel" in res['text']:
            found_tech_faiss = True
            
    if found_tech_faiss:
        print("SUCCESS: FAISS retrieved relevant documents.")
    else:
        print("FAILURE: FAISS did not retrieve expected documents.")
        
    print("\nPhase 3 Verification Complete.")

if __name__ == "__main__":
    test_phase3()
