import os
import pandas as pd
import pickle

df = pd.read_json("hf://datasets/gfissore/arxiv-abstracts-2021/arxiv-abstracts.jsonl.gz", lines=True)

os.makedirs("arxiv-abstracts-2021", exist_ok=True)
if os.path.exists("arxiv-abstracts-2021/pickle.pkl"):
    os.remove("arxiv-abstracts-2021/pickle.pkl")
with open("arxiv-abstracts-2021/pickle.pkl", "xb") as f:
    pickle.dump(df, f)