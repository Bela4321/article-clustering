from typing import List, Tuple, Any

import numpy as np
from numpy import ndarray, dtype
from torch.nn.functional import embedding
from transformers import AutoTokenizer, AutoModel
import torch

from embeddings.embedding_utils import simple_clean


def get_bert_embedding(text, model_name="bert-base-uncased"):
    """
    Generates an embedding for a given text using a pre-trained BERT model.

    Parameters:
        text (str): The input text to encode.
        model_name (str): The name of the pre-trained BERT model. Default is 'bert-base-uncased'.

    Returns:
        torch.Tensor: A tensor representing the text embedding.
    """
    # Load pre-trained BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")

    # Pass inputs through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden state
    last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

    # Get the mean pooling of the token embeddings (excluding [CLS] and [SEP] tokens)
    embedding = last_hidden_state.mean(dim=1)  # Shape: [batch_size, hidden_size]

    return embedding.squeeze()

def get_embedding(corpus: List[str], model_name="bert-base-uncased") -> tuple[ndarray[Any, dtype[Any]], None]:
    """
    Create a BERT embedding for a list of documents.

    Args:
        corpus: List of documents.
        model_name: Name of the pre-trained BERT model.

    Returns:
        BERT embeddings.
    """
    cleaned_corpus = simple_clean(corpus)
    embeddings = []
    for document in cleaned_corpus:
        embedding = get_bert_embedding(document, model_name)
        embeddings.append(embedding)
    return np.array(embeddings), None