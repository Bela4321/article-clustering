import os
import pickle
from typing import List


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



def load_arxiv_abstracts_2021(categorizer:Categorizer,querry=None,limit=1000) -> (List[str], List[List[int]]):
    with open(f"data/arxiv-abstracts-2021/pickle.pkl", "rb") as f:
        data = pickle.load(f)
        # cut data via querry
        #data = data_from_only_n_topics(data,2, limit)
        if querry:
            data = data.query(querry)[:limit]
        else:
            data = data[:limit]
        # string-concatenation of title and abstract
        content_data = data["title"] + " " + data["abstract"]
        categories = [category[0].split() for category in data["categories"]]
        numerical_catrgories = categorizer.fit_mulitlables(categories)

        return content_data, numerical_catrgories

def load_2_querries(categorizer:Categorizer,querry1,querry2,limit=1000) -> (List[str], List[List[int]]):
    with open(f"data/arxiv-abstracts-2021/pickle.pkl", "rb") as f:
        data = pickle.load(f)
        data = data.query(querry1)[:limit]
        data2 = data.query(querry2)[:limit]
        data = data.append(data2)
        # string-concatenation of title and abstract
        content_data = data["title"] + " " + data["abstract"]
        categories = [category[0].split() for category in data["categories"]]
        numerical_catrgories = categorizer.fit_mulitlables(categories)

        return content_data, numerical_catrgories