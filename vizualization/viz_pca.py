import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from vizualization.viz_utils import reduce_to_single_label


def plot_in_2d(embedding, labels=None, needs_reduction=False):
    """
    Reduce the embedding to 2D via pca and plot it.

    Args:
        embedding: Embedding of any shape.
        labels: Labels.
    """
    # if multiple labels are given, reduce to single labels
    if needs_reduction:
        labels = reduce_to_single_label(labels)


    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embedding)
    if labels is None:
        plt.scatter(reduced[:, 0], reduced[:, 1])
    else:
        for i in set(labels):
            plt.scatter(reduced[labels == i, 0], reduced[labels == i, 1], label=i)
    plt.show()