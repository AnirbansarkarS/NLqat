from core.base_pipeline import NLP
import time

def test_phase2():
    print("Initializing NLP (Phase 2 Verification)...")
    nlp = NLP()
    
    print("\n--- Testing Embeddings ---")
    text = "Artificial Intelligence is transforming the world."
    print(f"Embedding text: '{text}'")
    emb = nlp.embed(text)
    print(f"Embedding shape: {emb.shape}")
    assert emb.shape[0] > 0
    print("SUCCESS: Embedding generated.")

    print("\n--- Testing Similarity ---")
    t1 = "I love programming."
    t2 = "Coding is my passion."
    t3 = "I like eating pizza."
    
    sim1 = nlp.similarity(t1, t2)
    sim2 = nlp.similarity(t1, t3)
    
    print(f"Similarity ('{t1}', '{t2}'): {sim1:.4f}")
    print(f"Similarity ('{t1}', '{t3}'): {sim2:.4f}")
    
    if sim1 > sim2:
        print("SUCCESS: Semantic similarity logic holds.")
    else:
        print("FAILURE: Semantic similarity logic failed.")

    print("\n--- Testing Summarization ---")
    long_text = """
    Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. 
    Learning can be supervised, semi-supervised or unsupervised. 
    Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.
    """
    print("Summarizing text...")
    start = time.time()
    summary = nlp.summarize(long_text)
    print(f"Summary ({time.time()-start:.2f}s): {summary}")
    if len(summary) < len(long_text):
        print("SUCCESS: Summary is shorter than original.")

    print("\n--- Testing Clustering ---")
    sentences = [
        "The cat sits on the mat.",
        "Dogs are great pets.",
        "I love felines.",
        "Python is a programming language.",
        "Java code is verbose."
    ]
    print(f"Clustering {len(sentences)} sentences...")
    clusters = nlp.cluster(sentences)
    for cid, texts in clusters.items():
        print(f"Cluster {cid}: {texts}")
        
    print("\nPhase 2 Verification Complete.")

if __name__ == "__main__":
    test_phase2()
