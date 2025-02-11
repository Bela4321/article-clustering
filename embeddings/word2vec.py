import numpy as np
from typing import List, Tuple, Any
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize


def get_embedding(corpus: List[str]) -> tuple[Any, None]:
    """
    Given a corpus (list of strings), returns a Word2Vec embedding matrix.

    Each row in the matrix corresponds to a word vector for one word in the vocabulary.
    The ordering of the rows matches the ordering in model.wv.index_to_key.

    Args:
        corpus (List[str]): A list of texts (each text may contain one or more sentences).

    Returns:
        np.ndarray: A matrix of shape (vocab_size, vector_size) where each row is a word's embedding.
    """
    # Tokenize the corpus:
    # For each text in the corpus, break it into sentences and then tokenize each sentence into words.
    tokenized_sentences = []
    for text in corpus:
        for sentence in sent_tokenize(text):
            # Convert sentence to lowercase and tokenize into words
            tokens = word_tokenize(sentence.lower())
            tokenized_sentences.append(tokens)

    # Train a Word2Vec model using the tokenized sentences.
    # Here, min_count=1 ensures that even words that appear only once are included.
    # vector_size sets the dimensionality of the word vectors, and window defines the context window size.
    model = Word2Vec(tokenized_sentences, min_count=1, vector_size=100, window=5)

    document_embeddings = []
    for text in corpus:
        # Tokenize the text into words.
        tokens = word_tokenize(text.lower())
        word_vectors = []
        for token in tokens:
            # Check if the token exists in the Word2Vec model's vocabulary.
            if token in model.wv.key_to_index:
                word_vectors.append(model.wv[token])
        if word_vectors:
            # Average the word vectors to obtain the text-level embedding.
            avg_vector = np.mean(word_vectors, axis=0)
        else:
            # If no valid words are found (e.g., empty text), use a zero vector.
            avg_vector = np.zeros(model.vector_size)
        document_embeddings.append(avg_vector)

    # Convert the list of document embeddings to a NumPy array.
    embedding_matrix = np.vstack(document_embeddings)
    return embedding_matrix, None
