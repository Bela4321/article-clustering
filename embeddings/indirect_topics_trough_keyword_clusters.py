import numpy as np
from typing import List, Tuple

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from clustering.kmeans import xmeans_clustering
import umap
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import KMeans

import vizualization.viz_umap
from embeddings.embedding_utils import simple_clean_list





def load_translator() -> dict:
    """
    Lädt die FastText Vektoren.
    """
    filename = "data/vector_translator/translator.pickle"
    import pickle
    vectors = {}
    with open(filename, "rb") as f:
        vectors = pickle.load(f)
    return vectors

def get_embedding(corpus: List[str]) -> Tuple[np.ndarray, None]:
    """
    Erstellt Dokumenten-Embeddings
    """
    cleaned_corpus = simple_clean_list(corpus)

    # Sammle einzigartige Wörter aus dem Korpus
    word_freq = {}
    for document in cleaned_corpus:
        for word in document:
            word_freq[word] = word_freq.get(word, 0) + 1

    # filter keywords
    # select words based on frequency (between 10 and 90 percentile)
    sorted_words_by_freq = sorted(word_freq.items(), key=lambda x: x[1])
    upper_bound = int(len(sorted_words_by_freq) * 0.9)

    selected_words = {word for word, _ in sorted_words_by_freq[:upper_bound]}

    print("Loading FastText embeddings...")
    translator = load_translator()
    print("FastText embeddings loaded.")


    print("Creating keyword embeddings...")
    # get keyword embeddings
    keyword_list = list(set(translator.keys()).intersection(selected_words))
    keyword_embeddings = np.array([translator[keyword] for keyword in keyword_list])
    print("Keyword embeddings created.")



    # reduce keyword_embedding
    print("reduce dimensions of Keyword Embedding")
    umap_model = umap.UMAP(
        n_components=30)
    reduced_keyword_embedding = umap_model.fit_transform(keyword_embeddings)
    print("Keyword Embeddings reduced.")

    print("Clustering with kmeans")
    k=100
    kmeans = KMeans(n_clusters=k).fit(reduced_keyword_embedding)
    keyword_labels = np.array(kmeans.labels_, dtype=int)


    print("Clustering with xmeans...")
    k_start = min(4, reduced_keyword_embedding.shape[0])
    initial_centers = reduced_keyword_embedding[
        np.random.choice(reduced_keyword_embedding.shape[0], k_start, replace=False)]
    xmeans_instance = xmeans(reduced_keyword_embedding, initial_centers, kmax=25)
    xmeans_instance.process()

    keyword_labels = np.zeros(len(reduced_keyword_embedding), dtype=int)  # Assign cluster labels
    for cluster_idx, cluster in enumerate(xmeans_instance.get_clusters()):
        for idx in cluster:
            keyword_labels[idx] = cluster_idx

    keyword_cluster_centers = [np.mean(keyword_embeddings[np.array(keyword_labels) == lab], axis=0) for lab in range(max(keyword_labels) + 1)]
    print("Clustering with xmeans complete")

    document_embeddings = []

    cosine_cache = {}

    for document in cleaned_corpus:
        # calculate the similarity between every word and the keyword_cluster_centers
        embeddable_words = [word for word in document if word in translator]
        document_word_embeddings = np.array([translator[word] for word in document if word in translator])
        # remove the bias from the document embeddings

        # for every center, average of the top 10% are chosen as similarity
        keyword_cluster_coefficients = []
        for i, cluster_center in enumerate(keyword_cluster_centers):
            similarities = np.array(
                [cosine_cached(str(i) + "_" + word, cluster_center, word_embedd) for word, word_embedd in
                 zip(embeddable_words, document_word_embeddings)])

            cutoff = max(5, int(0.05 * len(similarities)))
            score = np.mean(np.sort(similarities)[-cutoff:])
            keyword_cluster_coefficients.append(score)

        document_embeddings.append(keyword_cluster_coefficients)

    final_document_embeddings = np.array(document_embeddings)

    # min-max scale each dimension to [-1,1]
    mins = np.min(final_document_embeddings, axis=0)
    maxs = np.max(final_document_embeddings, axis=0)
    final_document_embeddings = 2 * (final_document_embeddings - mins) / (maxs - mins) - 1

    # normalize
    final_document_embeddings = normalize(final_document_embeddings)

    return np.array(final_document_embeddings), None

cosine_cache = {}
def cosine_cached(key, cluster_center, word_embedding):
    if key in cosine_cache:
        return cosine_cache[key]
    sim = cosine_similarity(cluster_center.reshape(1, -1), word_embedding.reshape(1, -1))
    cosine_cache[key] = sim
    return sim



