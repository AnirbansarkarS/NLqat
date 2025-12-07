from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    # Ensure 2D arrays for sklearn
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
        
    return float(cosine_similarity(vec1, vec2)[0][0])

def semantic_search(query_vec: np.ndarray, corpus_vecs: np.ndarray, top_k: int = 5):
    """
    Finds the top_k most similar vectors in the corpus to the query vector.
    Returns indices and scores.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    
    similarities = cosine_similarity(query_vec, corpus_vecs)[0]
    # Get top_k indices sorted by score descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append((idx, similarities[idx]))
        
    return results
