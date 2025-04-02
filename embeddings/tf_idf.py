from typing import List, Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from umap import UMAP

from embeddings.embedding_utils import clean_and_stem


def term_document_matrix(corpus: List[str]) -> np.ndarray:
    """
    Create a term-document matrix from a list of documents.
    Args:
        corpus: List of documents.
    Returns:
        Term-document matrix.
    """
    cleaned_corpus = clean_and_stem(corpus)

    vectorizer = CountVectorizer()
    term_document = vectorizer.fit_transform(cleaned_corpus)
    return term_document.toarray()

def get_embedding(corpus: List[str]) -> tuple[Any, None]:
    """
    Create a TF-IDF matrix from a term-document matrix.

    Args:
        term_document: Term-document matrix.

    Returns:
        TF-IDF matrix.
        :param corpus:
    """
    cleaned_corpus = clean_and_stem(corpus)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(cleaned_corpus)
    return tfidf.toarray(), None


def get_embedding_pca(corpus: List[str]) -> Tuple[np.ndarray, None]:
    """
    Create a TF-IDF matrix from a term-document matrix and apply PCA to reduce to 100 dimensions.

    Args:
        term_document: Term-document matrix.

    Returns:
        TF-IDF matrix.
    """
    embedding, _ = get_embedding(corpus)
    from sklearn.decomposition import PCA
    dimensions = min(embedding.shape[0],embedding.shape[1], 100)
    print(dimensions)
    pca = PCA(n_components=dimensions)
    return pca.fit_transform(embedding), None

def get_embedding_UMAP(corpus):
    embedding, _ = get_embedding(corpus)
    from sklearn.decomposition import PCA
    dimensions = min(embedding.shape[0]-2, embedding.shape[1]-2, 100)
    print(dimensions)
    umap = UMAP(n_components=dimensions)
    return umap.fit_transform(embedding), None