import os

import numpy as np
import umap
from matplotlib import pyplot as plt

from data.Categorizer import Categorizer


def plot_in_2d(embedding, categorzier:Categorizer, labels=None, title:str= "UMAP Visualization",show_legend:bool=True):
    """
    Reduces the dimensionality of embeddings to 2D using UMAP and visualizes them.

    Parameters:
        embedding (np.ndarray): High-dimensional embeddings to visualize.
        labels (np.ndarray or list, optional): Cluster labels for coloring points. Default is None.

    Returns:
        None
        :param title: title displayed on the plot
        :param embedding: embedding matrix to visualize
        :param categorzier: Categorizer containing the labels
        :param labels: labels for each embedding
        :param needs_reduction: if True, reduce multiple labels to single labels
    """


    reducer = umap.UMAP(n_neighbors=25, min_dist=0., n_components=2,metric='cosine', random_state=42)

    # Reduce dimensionality
    reduced_embedding = reducer.fit_transform(embedding)

    # Plot
    plt.figure(figsize=(10, 8))
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if categorzier is not None:
                label_str = categorzier.get_label_str(label)
            else:
                label_str = label
            idx = labels == label
            plt.scatter(reduced_embedding[idx, 0], reduced_embedding[idx, 1], label=f"Cluster {label_str}", alpha=0.7)
        # Add legend
        if show_legend:
            plt.legend()
    else:
        plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], alpha=0.7)

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{title.replace(' ','_')}", dpi=300, bbox_inches='tight')
    plt.show()

def plot_fitting_mapping(embedding, categotizer, assigned_labels, true_labels):
    """

    :param embedding: Embedding matrix of the documents
    :param assigned_labels: Cluster labels assigned by the clustering algorithm
    :param true_labels: True labels of the documents (mulitlabels)
    :param categotizer: Categorizer object containing the label names

    As there are typlically more true labels than cluster labels, some true lables should be merged together.
    """
    assert len(assigned_labels) == len(true_labels)

    reduced_labels = np.zeros(len(assigned_labels))
    mergeMap = {}
    for assigned_label in set(assigned_labels):
        cluster = np.argmax(np.bincount(true_labels[assigned_labels == assigned_label]))
        reduced_labels[assigned_labels == assigned_label] = cluster
        if cluster not in mergeMap:
            mergeMap[cluster] = []
        mergeMap[cluster].append(assigned_label)



    plot_in_2d(embedding, categotizer, reduced_labels)