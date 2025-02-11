from data.Categorizer import Categorizer
from data.arxiv_abstracts_2021 import load_with_querry
import matplotlib.pyplot as plt


class DataExplorer:
    def __init__(self):
        self.data_loader = load_with_querry
        self.categorizer = Categorizer()
        self.data=None
        self.true_labels = None
        self.true_single_labels = None

    def load_data(self, base_query, additional_query=None, load_limit=None):
        full_query = f"abstract.str.contains('{base_query}')"
        if additional_query:
            full_query += f" and {additional_query}"
        print("Loading data...")
        if load_limit is None:
            self.data, self.true_labels = self.data_loader(self.categorizer, full_query)
        else:
            self.data, self.true_labels = self.data_loader(self.categorizer, full_query, load_limit)
        self.true_single_labels = self.categorizer.reduce_to_single_label(self.true_labels)
        print(f"Data loaded. {len(self.data)} entries.")

    def label_distribution(self, top_k=None, show_single_labels=False):
        # histogram of label distribution
        label_frequency = {}
        if show_single_labels:
            for label in self.true_single_labels:
                label_frequency[label] = label_frequency.get(label, 0) + 1
        else:
            for label_list in self.true_labels:
                for label in label_list:
                    label_frequency[label] = label_frequency.get(label, 0) + 1
        order = sorted(label_frequency.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            order = order[:top_k]
        labels = [str(self.categorizer.get_label_str(label[0])) for label in order]
        frequencies = [label[1] for label in order]
        plt.figure(figsize=(12, 6))
        plt.bar(labels, frequencies)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    explorer = DataExplorer()
    explorer.load_data("quantum")
    explorer.label_distribution(top_k=20,show_single_labels=True)