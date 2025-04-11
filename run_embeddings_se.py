from data.Categorizer import Categorizer
from multi_query_data.gateway_objects import load_dataset
from embeddings.embedding_utils import get_queries
from embeddings.tf_idf import get_embedding_pca as tf_idf_embedding_pca
from embeddings.tf_idf import get_embedding_UMAP as tf_idf_embedding_umap
from embeddings.fasttext import get_embedding_mean_polling as fasttext_mean_embedding
from embeddings.fasttext import get_embedding_max_polling as fasttext_max_embedding
from embeddings.fasttext import get_embedding_combined_polling_pca as fasttext_combined_embedding_pca
from embeddings.indirect_topics_trough_keyword_clusters import get_embedding_fresh_translator as keyword_embedding
from embeddings.openai_api import get_embedding as openai_embedding
from tqdm import tqdm

import os
import pickle

def get_data():
    #embedding_algos = [tf_idf_embedding_pca, fasttext_combined_embedding, openai_embedding][:-1]
    embedding_algos = [fasttext_combined_embedding_pca]
    data_loader = load_dataset
    return embedding_algos, data_loader



# load data with querries and create an embedding for each querry
if __name__ == "__main__":
    embedding_algos, data_loader = get_data()
    categorizer = Categorizer()

    for dataset_diff in ["easy", "medium", "hard"]:
        print(f"Caching data for {dataset_diff}")
        categorizer = Categorizer()
        data, numerical_labels = data_loader(categorizer, dataset_diff)
        print("data loaded")

        for embedding_algo in embedding_algos:

            dir_name = embedding_algo.__module__.split(".")[-1]+"__"+embedding_algo.__name__
            os.makedirs(f"embedding_results_se/{dir_name}", exist_ok=True)
            if os.path.exists(f"embedding_results_se/{dir_name}/{dataset_diff}.pkl"):
                continue

            print(f"creating embedding with {dir_name}")
            embeddings, clusters = embedding_algo(data)
            print(f"embedding created. {len(embeddings)} entries.")

            tosave = {
                "embeddings": embeddings,
                "clusters": clusters,
                "numerical_labels": numerical_labels,
                "categorizer": categorizer
            }

            if os.path.exists(f"embedding_results_se/{dir_name}/{dataset_diff}.pkl"):
                print("overwriting existing embedding results")
                os.remove(f"embedding_results_se/{dir_name}/{dataset_diff}.pkl")
            with open(f"embedding_results_se/{dir_name}/{dataset_diff}.pkl", "wb") as f:
                pickle.dump(tosave, f)
            print(f"embedding created and saved.\n")
