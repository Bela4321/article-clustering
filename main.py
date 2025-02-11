from SequentialClustering import SequentialClustering
from clustering.Kmeans import kmeans
from data.arxiv_abstracts_2021 import load_with_querry
from data.arxiv_abstracts_2021 import load_n_querries
from embeddings.bert import get_embedding as bert
from embeddings.n_dst import get_embedding as n_dst
from embeddings.word2vec import get_embedding as word2vec
from embeddings.arl_adv import get_embedding as arl_adv
from embeddings.btm import get_embedding as btm

if __name__ != "__main__":
    raise Exception("This module is not meant to be imported")

myClustering = SequentialClustering(
    data_loader=load_n_querries,
    embedding_method=word2vec,
    clustering_method=kmeans)

querries = ["quantum","social", "mathematics"]
myClustering.run(querries)