from typing import List
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


def hierarchical_clustering(data_matrix: np.array, k: int = 5) -> List[int]:
    """
    Performs hierarchical clustering on a given document embedding matrix and returns cluster labels.

    Parameters:
    data_matrix (np.array): A 2D numpy array where each row is an embedded document.
    k (int): The desired number of clusters. Default is 5.

    Returns:
    List[int]: A list of cluster labels assigned to each document.
    """
    # Perform hierarchical clustering
    linkage_matrix = linkage(data_matrix, method='ward')

    # Extract cluster labels
    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')

    return cluster_labels.tolist()