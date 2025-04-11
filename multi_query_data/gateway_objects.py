import pickle

from data.Categorizer import Categorizer


def load_dataset(categorizer:Categorizer, dataset_diff):
    datapath= f"multi_query_data/{dataset_diff}_combined_dataframe.pkl"
    with open(datapath, "rb") as f:
        data_df=pickle.load(f)
    content_data = list(data_df["text"])
    categories = data_df["source_query"]
    numerical_categories = categorizer.fit_singlelabels(categories)
    return content_data, numerical_categories