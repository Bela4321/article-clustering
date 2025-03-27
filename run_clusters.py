import pickle
import os

from vizualization.viz_umap import plot_in_2d
from clustering.kmeans import xmeans_clustering, hdbscan_clustering

def get_data():
    data_queries = {
        "Computer Science and AI": ["Transformer models", "Federated learning", "Quantum computing", "Explainable AI", "Graph neural networks"],
        "Physics and Engineering": ["Topological insulators", "Optical metamaterials", "Fission", "Soft robotics", "Health monitoring"],
        "Biology and Medicine": ["CRISPR", "Microbiome", "DNA sequencing", "Synthetic biology", "Drug delivery"],
        "Earth and Environmental Science": ["Climate model", "Remote sensing", "Greenhouse gas", "Biodiversity", "Light pollution"]
    }
    embedding_folder_path = "embedding_results/openai_api__get_embedding/"
    cluster_algos = [hdbscan_clustering]
    return data_queries, embedding_folder_path, cluster_algos

if __name__ == "__main__":
    data_queries, embedding_folder_path, cluster_algos = get_data()
    for category, data_queries in data_queries.items():
        for query in data_queries:
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
                save_path = f"cluster_results/{cluster_algo.__name__}/{query_key}.pkl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.exists(save_path):
                    print("overwriting existing clustering results")
                    os.remove(save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(tosave, f)

                #plot_in_2d(embedding, None, labels, f"{query_key} with {cluster_algo.__name__}", True)
