from SequentialClustering import SequentialClustering
from clustering.Kmeans import kmeans
from data.load_datasets import load_arxiv_abstracts_2021
from embeddings.bert import get_embedding as bert
from embeddings.n_dst import get_embedding as n_dst
from embeddings.arl_adv import get_embedding as arl_adv

if __name__ != "__main__":
    raise Exception("This module is not meant to be imported")

myClustering = SequentialClustering(
    data_loader=load_arxiv_abstracts_2021,
    embedding_method=arl_adv,
    clustering_method=kmeans)

myClustering.run("quantum",load_limit=1000)