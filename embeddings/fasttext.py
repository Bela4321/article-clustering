import io
import numpy as np
from typing import List, Tuple
from embeddings.embedding_utils import clean_and_stem_list


def load_vectors(fname: str, word_set: set) -> dict:
    """
    Lädt nur die Vektoren für die benötigten Wörter.
    """
    vectors = {}
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        for line in fin:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in word_set:
                vectors[word] = np.array([float(x) for x in tokens[1:]])
    return vectors


def get_embedding_mean_polling(corpus: List[str]) -> Tuple[np.ndarray, None]:
    """
    Erstellt Dokumenten-Embeddings
    """
    cleaned_corpus = clean_and_stem_list(corpus)

    # Sammle einzigartige Wörter aus dem Korpus
    unique_words = set()
    for document in cleaned_corpus:
        unique_words.update(document)

    print("Loading FastText embeddings...")
    translator = load_vectors("embeddings/fasttext_data/wiki-news-300d-1M.vec", unique_words)
    print("FastText embeddings loaded.")

    embeddings = []
    vector_size = 300
    for document in cleaned_corpus:
        word_vectors = []
        for word in document:
            if word in translator:
                word_vectors.append(translator[word])

        if word_vectors:
            document_embedding = np.mean(word_vectors, axis=0)
        else:
            document_embedding = np.zeros(vector_size)

        embeddings.append(document_embedding)

    return np.array(embeddings), None


def get_embedding_max_polling(corpus: List[str]) -> Tuple[np.ndarray, None]:
    """
        Erstellt Dokumenten-Embeddings
        """
    cleaned_corpus = clean_and_stem_list(corpus)

    # Sammle einzigartige Wörter aus dem Korpus
    unique_words = set()
    for document in cleaned_corpus:
        unique_words.update(document)

    print("Loading FastText embeddings...")
    translator = load_vectors("embeddings/fasttext_data/wiki-news-300d-1M.vec", unique_words)
    print("FastText embeddings loaded.")

    embeddings = []
    vector_size = 300
    for document in cleaned_corpus:
        word_vectors = []
        for word in document:
            if word in translator:
                word_vectors.append(translator[word])

        if word_vectors:
            document_embedding = np.max(word_vectors, axis=0)
        else:
            document_embedding = np.zeros(vector_size)

        embeddings.append(document_embedding)

    return np.array(embeddings), None

def get_embedding_min_polling(corpus: List[str]) -> Tuple[np.ndarray, None]:
    """
        Erstellt Dokumenten-Embeddings
        """
    cleaned_corpus = clean_and_stem_list(corpus)

    # Sammle einzigartige Wörter aus dem Korpus
    unique_words = set()
    for document in cleaned_corpus:
        unique_words.update(document)

    print("Loading FastText embeddings...")
    translator = load_vectors("embeddings/fasttext_data/wiki-news-300d-1M.vec", unique_words)
    print("FastText embeddings loaded.")

    embeddings = []
    vector_size = 300
    for document in cleaned_corpus:
        word_vectors = []
        for word in document:
            if word in translator:
                word_vectors.append(translator[word])

        if word_vectors:
            document_embedding = np.min(word_vectors, axis=0)
        else:
            document_embedding = np.zeros(vector_size)

        embeddings.append(document_embedding)

    return np.array(embeddings), None

def get_embedding_combined_polling(corpus: List[str]) -> Tuple[np.ndarray, None]:
    """
        Erstellt Dokumenten-Embeddings
        """
    cleaned_corpus = clean_and_stem_list(corpus)

    # Sammle einzigartige Wörter aus dem Korpus
    unique_words = set()
    for document in cleaned_corpus:
        unique_words.update(document)

    print("Loading FastText embeddings...")
    translator = load_vectors("embeddings/fasttext_data/wiki-news-300d-1M.vec", unique_words)
    print("FastText embeddings loaded.")

    embeddings = []
    vector_size = 300
    for document in cleaned_corpus:
        word_vectors = []
        for word in document:
            if word in translator:
                word_vectors.append(translator[word])

        if word_vectors:
            document_embedding_mean = np.mean(word_vectors, axis=0)
            document_embedding_max = np.max(word_vectors, axis=0)
            document_embedding_min = np.min(word_vectors, axis=0)
            document_embedding = np.concatenate((document_embedding_mean, document_embedding_max, document_embedding_min), axis=0)
        else:
            document_embedding = np.zeros(vector_size*3)

        embeddings.append(document_embedding)

    return np.array(embeddings), None