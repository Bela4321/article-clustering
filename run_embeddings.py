from data.Categorizer import Categorizer
from data.arxiv_abstracts_2021 import load_with_querry
from embeddings.embedding_utils import get_queries
from embeddings.tf_idf import get_embedding_pca as tf_idf_embedding_pca
from embeddings.tf_idf import get_embedding_UMAP as tf_idf_embedding_umap
from embeddings.fasttext import get_embedding_mean_polling as fasttext_mean_embedding
from embeddings.fasttext import get_embedding_max_polling as fasttext_max_embedding
from embeddings.fasttext import get_embedding_combined_polling as fasttext_combined_embedding
from embeddings.fasttext import get_embedding_combined_polling_pca as fasttext_combined_embedding_pca
from embeddings.indirect_topics_trough_keyword_clusters import get_embedding as keyword_embedding
from embeddings.openai_api import get_embedding as openai_embedding
from tqdm import tqdm

import os
import pickle

def get_data():
    data_querries = get_queries()
    embedding_algos = [tf_idf_embedding_pca, fasttext_mean_embedding, fasttext_max_embedding, fasttext_combined_embedding_pca, keyword_embedding, openai_embedding]
    embedding_algos = [tf_idf_embedding_pca]
    data_loader = load_with_querry
    return data_querries, embedding_algos, data_loader



# load data with querries and create an embedding for each querry
if __name__ == "__main__":
    data_querries, embedding_algos, data_loader = get_data()
    categorizer = Categorizer()

    for category, querries in tqdm(data_querries.items()):
        for querry in querries:
            query_key = (category[:5] + "_" + querry[:10]).replace(" ", "_")
            print(f"Is data for key {query_key} cached?")
            data_cache_path = f"query_results/{query_key}.pkl"
            os.makedirs("query_results", exist_ok=True)
            if os.path.exists(data_cache_path):
                print(f"Data cached for key {query_key}")
                with open(data_cache_path,"rb") as file:
                    data_dict = pickle.load(file)
                    data, numerical_labels, categorizer = data_dict["data"], data_dict["numerical_labels"], data_dict["categorizer"]
            else:
                print(f"Caching data for {query_key}")
                categorizer = Categorizer()
                data, numerical_labels = data_loader(categorizer, querry, limit=None)
                print("data loaded")
                to_save = {
                    "data": data,
                    "numerical_labels": numerical_labels,
                    "categorizer": categorizer
                }
                with open(data_cache_path, "wb") as f:
                    pickle.dump(to_save, f)
                print("data saved")




            for embedding_algo in embedding_algos:

                dir_name = embedding_algo.__module__.split(".")[-1]+"__"+embedding_algo.__name__
                os.makedirs(f"embedding_results/{dir_name}", exist_ok=True)
                if os.path.exists(f"embedding_results/{dir_name}/{query_key}.pkl"):
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

                if os.path.exists(f"embedding_results/{dir_name}/{query_key}.pkl"):
                    print("overwriting existing embedding results")
                    os.remove(f"embedding_results/{dir_name}/{query_key}.pkl")
                with open(f"embedding_results/{dir_name}/{query_key}.pkl", "wb") as f:
                    pickle.dump(tosave, f)
                print(f"embedding created and saved.\n")



