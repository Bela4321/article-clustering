from typing import List

import numpy as np
from sklearn.cluster import KMeans



def kmeans(data_matrix: np.array, k = 10)->List[int]:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_matrix)
    return kmeans.labels_