import os

import numpy as np
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


# Option 2: Direct assignment (be cautious not to expose your key)
# openai.api_key = "your_api_key_here"

def get_embedding(corpus: list[str]) -> [np.array, None]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding API and return a DataFrame.

    Args:
        corpus (list of str): List of short text strings.

    Returns:
        pd.DataFrame: DataFrame with columns 'text' and 'embedding'
    """
    # batch responses to 1000 docs each
    k= len(corpus)//1000+1
    corpus_batches = [corpus[low*1000:min((low+1)*1000,len(corpus))] for low in range(k)]
    # Call the embedding API (using the text-embedding-3-large model)
    all_embeddings = []
    for i in range(k):
        response = openai.embeddings.create(
            input=corpus_batches[i],
            model="text-embedding-3-large",
            dimensions=512
        )

        # Extract the embeddings from the response
        embeddings = [item.embedding for item in response.data]
        all_embeddings += embeddings

    # Create and return an array
    return np.array(all_embeddings), None
