import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class PatternClusterer:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_model = None
    
    def fit_predict(self, embeddings):
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(embeddings)
        return cluster_labels, self.kmeans_model
    
    def save_clusters(self, dates, labels, filename):
        df = pd.DataFrame({
            'Date': dates,
            'Cluster_ID': labels
        })
        df.to_csv(filename, index=False)
