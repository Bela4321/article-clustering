import re
from typing import List

from nltk import PorterStemmer
from nltk.corpus import stopwords


def simple_clean(corpus: List[str]) -> List[str]:
    """
    Make lowercase and remove special characters from a list of documents.
    :param corpus: List of documents.
    :return:  List of cleaned documents.
    """
    cleaned_corpus = []
    for document in corpus:
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)
        cleaned_corpus.append(cleaned_document)
    return cleaned_corpus

def clean_and_stem(corpus: List[str]) -> List[str]:
    """
    Clean a list of documents.

    Args:
        corpus: List of documents.

    Returns:
        Cleaned list word lists.
    """
    cleaned_corpus = []
    for document in corpus:
        cleaned_document_list = []
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)
        # remove stopwords and stem words
        stemmer = PorterStemmer()
        cleaned_document_list = [stemmer.stem(word) for word in cleaned_document.split() if word not in stopwords.words('english')]
        cleaned_corpus.append(" ".join(cleaned_document_list))
    return cleaned_corpus

def clean_and_stem_list(corpus: List[str]) -> List[List[str]]:
    """
    Clean a list of documents.

    Args:
        corpus: List of documents.

    Returns:
        Cleaned list word lists.
    """
    cleaned_corpus = []
    for document in corpus:
        cleaned_document_list = []
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)
        # remove stopwords and stem words
        stemmer = PorterStemmer()
        cleaned_document_list = [stemmer.stem(word) for word in cleaned_document.split() if word not in stopwords.words('english')]
        cleaned_corpus.append(cleaned_document_list)
    return cleaned_corpus


def get_k_highest_values(dictionary, k):
    """
    Get the k highest values from a dictionary.

    Args:
        dictionary: Dictionary of values.
        k: Number of highest values to return.

    Returns:
        List of k highest values.
    """
    return sorted(dictionary, key=dictionary.get, reverse=True)[:k]