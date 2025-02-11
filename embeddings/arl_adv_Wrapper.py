import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf

from embeddings.ARL_Adv.arl import ARL


class ARL_WRAPPER:
    def __init__(self, corpus, emb_dim=50, num_clusters=5, neg_num=2, lr=1e-3, epochs=3):
        self.ARL = None
        self.corpus = corpus
        self.tokenized_corpus = None
        self.emb_dim = emb_dim
        self.num_clusters = num_clusters
        self.neg_num = neg_num
        self.lr = lr
        self.epochs = epochs
        self.session = None


    def save_arl(self, path):
        self.ARL.save(path)

    def load_arl(self, path):
        self.ARL = ARL.load(path)

    def init_arl(self):
        self.tokenized_corpus, vocab_size = ARL_WRAPPER.tokenize_corpus(self.corpus)
        np.random.seed(123)
        w_embed_init = (np.random.rand(vocab_size, self.emb_dim) - 0.5) / self.emb_dim
        c_embed_init = (np.random.rand(self.num_clusters, self.emb_dim) - 0.5) / self.emb_dim

        tf.compat.v1.reset_default_graph()
        self.session = tf.compat.v1.Session()
        self.ARL = ARL({"ns": self.neg_num, "lr": self.lr})
        self.ARL.set_session(self.session)
        self.ARL.build(w_embed_init.astype(np.float32), c_embed_init.astype(np.float32))
        self.session.run(tf.compat.v1.global_variables_initializer())





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