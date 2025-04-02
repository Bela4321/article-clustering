import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_completeness_v_measure


def sort_confusion_matrix(confusion_matrix):
    """
    Sort columns and rows so that the diagonal has the highest values.
    :param confusion_matrix:
    :return:
    """
    min_dim = min(confusion_matrix.shape )
    for i in range(min_dim):
        # find the maximum value in matrix and move it to (i,i)
        max_value = np.max(confusion_matrix[i:, i:])
        max_index = np.where(confusion_matrix == max_value)
        max_index = (max_index[0][0], max_index[1][0])

        # swap the rows
        confusion_matrix[[i, max_index[0]]] = confusion_matrix[[max_index[0], i]]
        # swap the columns
        confusion_matrix[:, [i, max_index[1]]] = confusion_matrix[:, [max_index[1], i]]


    return confusion_matrix


def internal_evaluation(embeddings, assigned_labels):
    """
    :arg embeddings: Embedding matrix of the documents
    :arg assigned_labels: Cluster labels assigned by the clustering algorithm
    """
    # Silhouette score
    silhouette = silhouette_score(embeddings, assigned_labels)
    print(f"Silhouette score: {silhouette}")

    # Davies-Bouldin score
    davies_bouldin = davies_bouldin_score(embeddings, assigned_labels)
    print(f"Davies-Bouldin score: {davies_bouldin}")

    # Calinski-Harabasz score
    calinski_harabasz = calinski_harabasz_score(embeddings, assigned_labels)
    print(f"Calinski-Harabasz score: {calinski_harabasz}")
    return silhouette, davies_bouldin, calinski_harabasz


def external_evaluation(assigned_labels, true_labels):
    """
    :arg assigned_labels: Cluster labels assigned by the clustering algorithm
    :arg true_labels: True labels of the documents
    """
    confusion_matrix



def full_evaluation(embeddings, assigned_labels, true_labels):
    """
    :arg embeddings: Embedding matrix of the documents
    :arg assigned_labels: Cluster labels assigned by the clustering algorithm
    :arg true_labels: True labels of the documents
    """
    # confusion matrix
    plot_confusion_matrix(assigned_labels, true_labels)

    internal_evaluation(embeddings, assigned_labels)
    external_evaluation(assigned_labels, true_labels)

