from typing import List

from embeddings.ARL_Adv.arl import ARL
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

# Enable TF 1.x compatibility mode
tf.compat.v1.disable_eager_execution()

def tokenize_corpus(corpus, pretrained_model="allenai/scibert_scivocab_uncased"):
    """
    Tokenize the given corpus using a pre-trained model.
    :param corpus: List of documents.
    :param pretrained_model: Name of the pre-trained model.
    :return: List of tokenized documents.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenized = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
    vocab_size = tokenizer.vocab_size
    return tokenized, vocab_size

def generate_negative_samples(tokenized_corpus, neg_num=5):
    """
    Randomly shuffle the corpus and select negative examples.
    Works with a tokenized corpus from a pretrained tokenizer.
    Args:
        tokenized_corpus: Tokenized output from a pretrained tokenizer (e.g., dictionary with 'input_ids').
        neg_num: Number of negative samples to generate.
    Returns:
        A list of 'neg_num' lists, each containing shuffled token ID sequences.
    """
    # Extract the token IDs (input_ids) from the tokenized corpus
    input_ids_corpus = tokenized_corpus["input_ids"].tolist()  # Convert to list for easier manipulation

    # Get all indices for shuffling
    all_indices = list(range(len(input_ids_corpus)))
    neg_batches = []

    for _ in range(neg_num):
        np.random.shuffle(all_indices)  # Shuffle indices
        # Create a shuffled batch based on the shuffled indices
        neg_batch = [input_ids_corpus[i] for i in all_indices]
        neg_batches.append(neg_batch)

    return neg_batches


def get_embedding(corpus: List[str]):
    corpus = list(corpus)
    doc_embedding, clusters = get_embeddings(corpus)
    return doc_embedding, clusters

def get_embeddings(corpus, emb_dim=50, num_clusters=5, neg_num=2, lr=1e-3, epochs=3):
    """
    Build an ARL model, train on the given corpus, and return cluster assignments.
    """
    # 1) Build vocabulary and tokenize the corpus
    tokenized_corpus, vocab_size = tokenize_corpus(corpus)

    # 2) Initialize embeddings (randomly for demo)
    np.random.seed(123)
    w_embed_init = (np.random.rand(vocab_size, emb_dim) - 0.5) / emb_dim
    c_embed_init = (np.random.rand(num_clusters, emb_dim) - 0.5) / emb_dim

    # 3) Create ARL model
    args = {'ns': neg_num, 'lr': lr}
    model = ARL(args)

    # 4) Reset graph & build the model under tf.compat.v1
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    model.set_session(sess)
    model.build(w_embed_init.astype(np.float32),
                c_embed_init.astype(np.float32))

    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # 5) Train the model for a few epochs
    for epoch in range(epochs):
        neg_batches = generate_negative_samples(tokenized_corpus, neg_num=neg_num)

        # Extract token IDs from the tokenized corpus
        input_ids_corpus = tokenized_corpus["input_ids"].tolist()  # Convert tensor to list for compatibility

        # ARL expects a list of objects with a .tokenids attribute
        # We'll create objects with lambdas that hold token IDs
        pos_batch = [lambda x=seq: None for seq in input_ids_corpus]
        for i, seq in enumerate(input_ids_corpus):
            pos_batch[i].tokenids = seq

        # Process negative batches similarly
        neg_batch_list = []
        for nb in neg_batches:  # Convert tensor to list
            # Create list of objects with .tokenids
            nb_docs = [lambda x=seq: None for seq in nb]
            for i, seq in enumerate(nb):
                nb_docs[i].tokenids = seq
            neg_batch_list.append(nb_docs)

        # Single training step on this "batch"
        model.train_step(pos_batch, neg_batch_list)

    # 6) Get cluster assignments
    pos_batch = [lambda x=seq: None for seq in input_ids_corpus]
    for i, seq in enumerate(input_ids_corpus):
        pos_batch[i].tokenids = seq

    clusters, doc_embeddings = model.predict(pos_batch)

    sess.close()

    return doc_embeddings, clusters

if __name__ == "__main__":
    corpus = [
        "cat sat on the mat",
        "dog chased the cat",
        "cat chased the mouse",
        "the dog sat on a log",
        "another random sentence",
    ]

    doc_embedding, cluster_assignments = get_embeddings(
        corpus, emb_dim=10, num_clusters=3, neg_num=2, epochs=5
    )
    print("Cluster assignments:", cluster_assignments)
