import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from vizualization.viz_utils import reduce_to_single_label


def plot_in_2d(embedding, labels=None, needs_reduction=False):
    """
    Reduces the dimensionality of embeddings to 2D using t-SNE and visualizes them.

    Parameters:
        embedding (np.ndarray): High-dimensional embeddings to visualize.
        labels (np.ndarray or list, optional): Cluster labels for coloring points. Default is None.

    Returns:
        None
    """
    # if multiple labels are given, reduce to single labels
    if needs_reduction:
        labels = reduce_to_single_label(labels)

    reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

    # Reduce dimensionality
    reduced_embedding = reducer.fit_transform(embedding)

    # Plot
    plt.figure(figsize=(10, 8))
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            plt.scatter(reduced_embedding[idx, 0], reduced_embedding[idx, 1], label=f"Cluster {label}", alpha=0.7)
    else:
        plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], alpha=0.7)

    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
