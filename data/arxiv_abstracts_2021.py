import os
import pickle
from typing import List

import pandas as pd

from data.Categorizer import Categorizer


def data_from_only_n_topics(data, n, limit):
    # select most prominent topics within limit
    topics = data[:limit]["categories"]
    topic_count= {}
    for topic_list in topics:
        for topic in topic_list:
            topic_count[topic]= topic_count.get(topic,0)+1
    most_common_topics = sorted(topic_count.items(), key=lambda x: x[1], reverse=True)[:n]
    most_common_topics = [topic[0] for topic in most_common_topics]
    data = data[[any([topic in topics for topic in most_common_topics]) for topics in data["categories"]]][:limit]
    return data



def load_with_querry_anymatch(categorizer:Categorizer, querry=None, limit=1000) -> (List[str], List[List[int]]):
    with open(f"data/arxiv-abstracts-2021/pickle.pkl", "rb") as f:
        data = pickle.load(f)
        # cut data via querry
        #data = data_from_only_n_topics(data,2, limit)
        if querry:
            words = querry.split()
            query_str = " & ".join([f"abstract.str.contains('{word}')" for word in words])
            if limit:
                data = data.query(query_str)[:limit]
            else:
                data = data.query(query_str)
        else:
            data = data[:limit]
        # string-concatenation of title and abstract
        content_data = data["title"] + " " + data["abstract"]
        categories = [category[0].split() for category in data["categories"]]
        numerical_catrgories = categorizer.fit_mulitlables(categories)

        return content_data, numerical_catrgories

def load_with_querry(categorizer:Categorizer, querry=None, limit=1000) -> (List[str], List[List[int]]):
    with open(f"data/arxiv-abstracts-2021/pickle.pkl", "rb") as f:
        data = pickle.load(f)
        # cut data via querry
        #data = data_from_only_n_topics(data,2, limit)
        if querry:
            if limit:
                data = data.query(f"abstract.str.contains('{querry}', case=False)")[:limit]
            else:
                data = data.query(f"abstract.str.contains('{querry}', case=False)")
        else:
            data = data[:limit]
        # string-concatenation of title and abstract
        content_data = data["title"] + "\n" + data["abstract"]
        categories = [category[0].split() for category in data["categories"]]
        numerical_catrgories = categorizer.fit_mulitlables(categories)

        return content_data, numerical_catrgories


def load_n_querries(categorizer:Categorizer, querries:List, limit=100) -> (List[str], List[List[int]]):
    '''
    Load data for each querry and return the combined data, the querry keywords are assigned as categories
    :param categorizer:
    :param querries:
    :param limit:
    :return:
    '''
    with open(f"data/arxiv-abstracts-2021/pickle.pkl", "rb") as f:
        data = pickle.load(f)
        data_df = pd.DataFrame()
        for querry in querries:
            querry_data = data.query(f"abstract.str.contains('{querry}')")[:limit]
            querry_data["categories"] = querry
            data_df = pd.concat([data_df,querry_data])
        # string-concatenation of title and abstract
        content_data = data_df["title"] + " " + data_df["abstract"]
        # limit to 1000 characters
        content_data = [content[:1000] for content in content_data]
        categories = [category for category in data_df["categories"]]
        numerical_catrgories = categorizer.fit_lables(categories)

        return content_data, numerical_catrgories


def load_training_data(val_size=10000):
    categorizer = Categorizer()
    with open(f"data/arxiv-abstracts-2021/pickle.pkl", "rb") as f:
        data = pickle.load(f)
        data_df = pd.DataFrame()
        data = data[val_size:]
        # string-concatenation of title and abstract
        content_data = data["title"] + " " + data["abstract"]
        categories = [category[0].split() for category in data["categories"]]
        numerical_catrgories = categorizer.fit_lables(categories)

        return content_data, numerical_catrgories