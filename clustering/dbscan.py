from typing import List

import numpy as np
from sklearn.cluster import DBSCAN
from sympy.abc import epsilon


def dbscan(data_matrix: np.array, k = 10)->List[int]:
    eps = estimate_epsilon(data_matrix)
    dbscan_model = DBSCAN(eps=eps, min_samples=10).fit(data_matrix)
    return dbscan_model.labels_



def estimate_epsilon(data_matrix: np.array, min_samples:int = 10)->float:
    """
    Estimate the epsilon parameter for DBSCAN with annoy/knn by finding the elbow in the k-distance graph.
    :param data_matrix:
    :param min_samples:
    :return:
    """
    from annoy import AnnoyIndex
    import matplotlib.pyplot as plt
    import numpy as np

    # Build the annoy index
    f = data_matrix.shape[1]
    t = AnnoyIndex(f, 'angular')
    for i in range(data_matrix.shape[0]):
        t.add_item(i, data_matrix[i])
    t.build(10)

    # Calculate the k-distance for each point
    k_distances = []
    for i in range(data_matrix.shape[0]):
        distances = t.get_nns_by_item(i, min_samples)
        k_distances.append(distances[-1])

    # Sort the distances
    k_distances = np.sort(k_distances)

    # Calculate the gradient
    gradients = []
    for i in range(1, len(k_distances)):
        gradients.append(k_distances[i] - k_distances[i-1])

    # Find the elbow
    elbow_index = np.argmax(gradients)
    epsilon = k_distances[elbow_index]

    # Plot the graph
    plt.plot(k_distances)
    plt.title("K-Distance Graph")
    plt.xlabel("Data Point")
    plt.ylabel("K-Distance")
    plt.show()
    print(f"Estimated epsilon: {epsilon}")
    return epsilon