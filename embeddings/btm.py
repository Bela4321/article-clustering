import bitermplus as btm
import numpy as np
import pandas as pd
import tmplot
from ipywidgets.embed import embed_minimal_html
import webbrowser
import panel as pn

def view_report(report_widget):
    pn.extension('ipywidgets', 'bokeh')
    pane = pn.panel(report_widget)
    # Serve the panel as a web app on a local server
    pane.show()

def get_embedding(corpus: list[str]):
    """
    Create a Biterm Topic Model (BTM) embedding for a list of documents.

    Args:
        corpus: List of documents.

    Returns:
        Biterm Topic Model (BTM) embeddings.
        Biterm topic model clusters.
    """
    # Create a Biterm Topic Model (BTM) instance
    texts = list(corpus)
    # Preprocess the corpus
    X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    tf = np.array(X.sum(axis=0)).ravel()
    # Vectorizing documents
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    docs_lens = list(map(len, docs_vec))
    # Generating biterms
    biterms = btm.get_biterms(docs_vec)

    # INITIALIZING AND RUNNING MODEL
    model = btm.BTM(
        X, vocabulary, seed=12321, T=8, M=20, alpha=50 / 8, beta=0.01)
    model.fit(biterms, iterations=20)
    p_zd = model.transform(docs_vec)

    return p_zd, model.labels_