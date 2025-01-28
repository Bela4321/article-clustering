import numpy as np
import umap
from matplotlib import pyplot as plt

from data.Categorizer import Categorizer
from vizualization.viz_utils import reduce_to_single_label


def plot_in_2d(embedding, categorzier:Categorizer, labels=None, needs_reduction=False):
    """
    Reduces the dimensionality of embeddings to 2D using UMAP and visualizes them.

    Parameters:
        embedding (np.ndarray): High-dimensional embeddings to visualize.
        labels (np.ndarray or list, optional): Cluster labels for coloring points. Default is None.

    Returns:
        None
        :param embedding: embedding matrix to visualize
        :param categorzier: Categorizer containing the labels
        :param labels: labels for each embedding
        :param needs_reduction: if True, reduce multiple labels to single labels
    """
    # if multiple labels are given, reduce to single labels
    if needs_reduction:
        labels = reduce_to_single_label(labels)


    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)

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
        plt.legend()
    else:
        plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], alpha=0.7)

    plt.title("UMAP Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

