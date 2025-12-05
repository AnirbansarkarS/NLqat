from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict

class Clusterer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def cluster(self, embeddings: np.ndarray, texts: List[str]) -> Dict[int, List[str]]:
        """
        Clusters the provided embeddings and returns a dictionary mapping cluster ID to list of texts.
        """
        if len(texts) < self.n_clusters:
            # Adjust n_clusters if we have fewer documents than requested clusters
            self.model.n_clusters = len(texts)
            
        labels = self.model.fit_predict(embeddings)
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx])
            
        return clusters
