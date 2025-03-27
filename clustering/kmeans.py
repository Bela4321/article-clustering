from typing import List
import numpy as np
from sklearn.cluster import KMeans
from pyclustering.cluster.xmeans import xmeans
import hdbscan
from diptest import diptest
from pyclustering.cluster.gmeans import gmeans
from clustpy.partition.dipmeans import DipMeans


def kmeans(data_matrix: np.array, k = 5)->List[int]:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_matrix)
    return kmeans.labels_

def xmeans_clustering(data_matrix: np.array) -> [List[int],np.array]:
    """
    X-Means clustering: Automatically estimates the number of clusters.
    """
    k_start = min(1, data_matrix.shape[0])
    initial_centers = data_matrix[np.random.choice(data_matrix.shape[0], k_start, replace=False)]
    xmeans_instance = xmeans(data_matrix, initial_centers)
    xmeans_instance.process()

    labels = np.zeros(len(data_matrix), dtype=int)  # Assign cluster labels
    for cluster_idx, cluster in enumerate(xmeans_instance.get_clusters()):
        for idx in cluster:
            labels[idx] = cluster_idx

    centroids = xmeans_instance.get_centers()

    return labels.tolist(),np.array(centroids)

def gmeans_clustering(data_matrix: np.array) -> List[int]:
    """
    G-Means clustering: Iteratively splits clusters based on Gaussian testing.
    """
    gmeans_instance = gmeans(data_matrix, tolerance=10)
    gmeans_instance.process()
    return gmeans_instance.get_clusters()


def hdbscan_clustering(data_matrix: np.array) -> List[int]:
    """
    HDBSCAN clustering: Finds clusters with variable densities and automatically determines k.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30,min_samples = 1)
    labels = clusterer.fit_predict(data_matrix)
    return labels.tolist(), None


def dipmeans_clustering(data_matrix: np.array) -> List[int]:
    """
    Dip-Means clustering: Uses Hartigan’s Dip Test to find multimodal clusters.
    """
    dipmeans = DipMeans()
    dipmeans.fit(data_matrix)
    return dipmeans.labels_


def from_clusters_to_labels(clusters:List[List[int]])->List[int]:
    pass

