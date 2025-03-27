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


def plot_confusion_matrix(assigned_labels, true_labels):
    """
    Plot a confusion matrix.

    Args:
        :param true_labels:
        :param assigned_labels:
    """

    # Get the number of clusters
    n_clusters = len(set(assigned_labels))
    # Get the number of labels
    n_labels = len(set(true_labels))

    map_true_label_to_position = {label: i for i, label in enumerate(set(true_labels))}


    confusion_matrix = np.zeros((n_clusters, n_labels))

    for i, cluster in enumerate(assigned_labels):
        confusion_matrix[cluster, map_true_label_to_position[true_labels[i]]] += 1

    #sort the confusion matrix, so diagonal is the highest
    confusion_matrix = sort_confusion_matrix(confusion_matrix)

    plt.imshow(confusion_matrix, cmap='viridis')

    plt.colorbar()
    plt.xlabel('True label')
    plt.ylabel('Cluster')
    plt.show()

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


def external_evaluation(assigned_labels, true_labels):
    """
    :arg assigned_labels: Cluster labels assigned by the clustering algorithm
    :arg true_labels: True labels of the documents
    """
    # Adjusted Rand index
    adjusted_rand = adjusted_rand_score(true_labels, assigned_labels)
    print(f"Adjusted Rand index: {adjusted_rand}")

    # Normalized mutual information
    nmi = normalized_mutual_info_score(true_labels, assigned_labels)
    print(f"Normalized mutual information: {nmi}")

    # Homogeneity, completeness, V-measure
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, assigned_labels)
    print(f"Homogeneity: {homogeneity}")
    print(f"Completeness: {completeness}")
    print(f"V-measure: {v_measure}")



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

