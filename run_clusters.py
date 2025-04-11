import pickle
import os

from embeddings.embedding_utils import get_queries
from vizualization.viz_umap import plot_in_2d
from clustering.clusterings import kmeans_with_estimated_k,xmeans_clustering, hdbscan_clustering, agglomerative_clustering_with_estimated_k
from embeddings.tf_idf import get_embedding_pca
from embeddings.openai_api import get_embedding as embedding_openai
from embeddings.fasttext import get_embedding_combined_polling_pca as fasttext_combined_pooling

if __name__ == "__main__":
    data_queries = get_queries()
    embeddings = [get_embedding_pca,fasttext_combined_pooling,embedding_openai]
    cluster_algos = [kmeans_with_estimated_k, xmeans_clustering, hdbscan_clustering,
                     agglomerative_clustering_with_estimated_k]
    cluster_algos = [hdbscan_clustering]

    for embedding in embeddings:
        chosen_embedding = embedding.__module__.split(".")[-1]+"__"+embedding.__name__
        print("\n",chosen_embedding)
        embedding_folder_path = f"embedding_results/{chosen_embedding}/"

        for category, queries in data_queries.items():
            for query in queries:
                query_key = (category[:5] + "_" + query[:10]).replace(" ", "_")
                print(f"loading embedding for {query}...")
                data_path = embedding_folder_path + query_key + ".pkl"
                with open(data_path,"rb") as file:
                    data_dict = pickle.load(file)
                embedding, numerical_labels, categorizer = data_dict["embeddings"], data_dict["numerical_labels"], data_dict["categorizer"]
                print(f"embedding loaded with shape: {embedding.shape}")
                for cluster_algo in cluster_algos:
                    print(f"running clustering with {cluster_algo.__name__}")
                    labels, centroids = cluster_algo(embedding)
                    tosave = {
                        "embeddings": embedding,
                        "clusters": labels,
                        "centroids": centroids,
                        "numerical_labels_true": numerical_labels,
                        "categorizer": categorizer
                    }
                    save_path = f"cluster_results/{chosen_embedding}/{cluster_algo.__name__}/{query_key}.pkl"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if os.path.exists(save_path):
                        print("overwriting existing clustering results")
                        os.remove(save_path)
                    with open(save_path, "wb") as f:
                        pickle.dump(tosave, f)

