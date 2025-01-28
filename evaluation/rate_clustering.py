import numpy as np


def plot_confusion_matrix(confusion_matrix):
    """
    Plot a confusion matrix.

    Args:
        confusion_matrix: Confusion matrix.
    """
    import matplotlib.pyplot as plt
    plt.imshow(confusion_matrix, cmap='viridis')
    plt.colorbar()
    plt.xlabel('True label')
    plt.ylabel('Cluster')
    plt.show()


def rate_from_multilabels(clustering, labels):
    """
    :arg clustering: list of cluster labels
    :arg labels: list of true (multi-)labels

    """
    # Get the number of clusters
    n_clusters = len(set(clustering))
    # Get the number of labels
    n_labels = len(set([label for label_list in labels for label in label_list]))
    # Create a confusion matrix
    confusion_matrix = np.zeros((n_clusters, n_labels))
    for i, cluster in enumerate(clustering):
        for label in labels[i]:
            confusion_matrix[cluster, label] += 1
    # Calculate the purity
    purity = np.sum(np.max(confusion_matrix, axis=0)) / len(clustering)
    plot_confusion_matrix(confusion_matrix)
    return purity

