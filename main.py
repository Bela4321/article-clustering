import sys

from SequentialClustering import SequentialClustering
from clustering.clusterings import kmeans
from data.arxiv_abstracts_2021 import load_with_querry
from data.arxiv_abstracts_2021 import load_n_querries
from embeddings.bert import get_embedding as bert
from embeddings.n_dst import get_embedding as n_dst
from embeddings.word2vec import get_embedding as word2vec
from embeddings.fasttext import get_embedding as fasttext
from embeddings.minillm_transformer import get_embedding as minillm_transformer
from embeddings.fasttext import get_embedding_idf_scale as fasttext_idf
from embeddings.btm import get_embedding as btm
from embeddings.openai_api import get_embedding as openai_api
from embeddings.indirect_topics_trough_keyword_clusters import get_embedding as indirect_topics

if __name__ == "__main__":

    myClustering = SequentialClustering(
        data_loader=load_n_querries,
        embedding_method=indirect_topics,
        clustering_method=kmeans)

    querries = ["machine learning", "quantum computing", "cryptography", "neural networks", "social networks"]
    myClustering.run(querries)