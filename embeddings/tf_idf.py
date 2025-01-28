from typing import List, Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

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


