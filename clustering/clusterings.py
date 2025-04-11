import math
from typing import List, Any
import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from pyclustering.cluster.xmeans import xmeans
from diptest import diptest
from pyclustering.cluster.gmeans import gmeans
from clustpy.partition.dipmeans import DipMeans
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP

from embeddings.embedding_utils import optimal_pca


def kmeans(data_matrix: np.array, k)->List[int]:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_matrix)
    return kmeans.labels_

def kmeans_with_estimated_k(data_matrix: np.array)->List[int]:
    estimated_k=approximate_k(data_matrix.shape[0])
    kmeans = KMeans(n_clusters=estimated_k, random_state=0).fit(data_matrix)
    return kmeans.labels_, None

def xmeans_clustering(data_matrix: np.array) -> [List[int],np.array]:
    """
    X-Means clustering: Automatically estimates the number of clusters.
    """
    k_start = min(3, data_matrix.shape[0])
    initial_centers = data_matrix[np.random.choice(data_matrix.shape[0], k_start, replace=False)]
    xmeans_instance = xmeans(data_matrix, initial_centers)
    xmeans_instance.process()

    labels = np.zeros(len(data_matrix), dtype=int)  # Assign cluster labels
    for cluster_idx, cluster in enumerate(xmeans_instance.get_clusters()):
        for idx in cluster:
            labels[idx] = cluster_idx

    centroids = xmeans_instance.get_centers()

    return labels.tolist(),np.array(centroids)

def hdbscan_clustering(data_matrix: np.array) -> tuple[Any, None]:
    """
    HDBSCAN clustering: Finds clusters with variable densities and automatically determines k.
    """
    #preprpcessing
    reduced_data_matrix = optimal_pca(data_matrix,0.4)
    clusterer = HDBSCAN(max_cluster_size=80, min_cluster_size=5,min_samples = 1)
    labels = clusterer.fit_predict(reduced_data_matrix)
    return labels.tolist(), None

def agglomerative_clustering_with_estimated_k(data_matrix: np.array) -> tuple[Any, None]:
    estimated_k = approximate_k(data_matrix.shape[0])
    agg = AgglomerativeClustering(n_clusters=estimated_k)
    return agg.fit_predict(data_matrix), None

def approximate_k(n):
    return int(min(max(math.sqrt(n) // 4, 3),n))

