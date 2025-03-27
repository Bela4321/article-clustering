from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def get_embedding(corpus: List[str]) -> [np.array, None]:

    corpus = list(corpus)
    # Modell und Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # tokenize corpus
    encoded_input = tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')

    # Embeddings berechnen
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Mean Pooling
    token_embeddings = model_output[0]
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                    min=1e-9)

    # Normalisierung
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.numpy(), None