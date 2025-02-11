import time
from typing import List, Union

from data.Categorizer import Categorizer
from vizualization.viz_umap import plot_in_2d as umap_plot
from vizualization.viz_umap import plot_fitting_mapping
from evaluation.rate_clustering import full_evaluation

class SequentialClustering:
    def __init__(self, data_loader, embedding_method, clustering_method, rating_method= full_evaluation, visualization_method_1 = umap_plot, visualization_method_2 = plot_fitting_mapping):
        """
        Initializes the SequentialClustering pipeline with dependency injection.

        Parameters:
            data_loader (function): Function to load data.
            embedding_method (function): Function to create embeddings from data.
            clustering_method (function): Function to perform clustering.
            rating_method (function): Function to rate clustering performance.
            visualization_method_1 (function): Function to visualize clusters.
        """
        self.data_loader = data_loader
        self.embedding_method = embedding_method
        self.clustering_method = clustering_method
        self.rating_method = rating_method
        self.visualization_method_1 = visualization_method_1
        self.visualization_method_2 = visualization_method_2

        self.categorizer = Categorizer()
        # store intermediate results
        self.data = None
        self.true_labels = None
        self.true_single_labels = None
        self.embeddings = None
        self.assigned_labels = None

    def run(self, query: Union[str,List[str]]):
        """
        Executes the sequential clustering pipeline.

        Parameters:
            base_query (str): Base query string to filter or load specific data (e.g., 'quantum').
            reduce_labels (function, optional): Function to reduce multi-labels to single labels for visualization.
            additional_query (str, optional): Additional query parameter for data filtering. Default is None.

        Returns:
            None
            :param base_querry_2:
            :param base_query:
        """
        # Load data
        print("Loading data...")
        start_time = time.time()
        if isinstance(query, list):
            self.data, self.true_labels = self.data_loader(self.categorizer, query)
        else:
            self.data, self.true_labels = self.data_loader(self.categorizer, query)
        self.true_single_labels = self.categorizer.reduce_to_single_label(self.true_labels)
        print(f"Data loaded. {len(self.data)} entries. Time taken: {time.time() - start_time:.2f} seconds.")

        # Create embeddings
        print("Creating embeddings...")
        start_time = time.time()
        self.embeddings, self.assigned_labels = self.embedding_method(self.data)
        print(f"Embeddings created with shape {self.embeddings.shape}. Time taken: {time.time() - start_time:.2f} seconds.")

        if self.assigned_labels is None:
            # Perform clustering
            print("Performing clustering...")
            start_time = time.time()
            self.assigned_labels = self.clustering_method(self.embeddings)
            print(f"Clustering done. Time taken: {time.time() - start_time:.2f} seconds.")
        else:
            print("Clustering labels provided. Skipping clustering step.")

        if self.rating_method is not None:
            # Rate clustering
            print("Rating clustering...")
            start_time = time.time()
            purity = self.rating_method(self.embeddings, self.assigned_labels, self.true_single_labels)
            print(f"Purity: {purity}. Time taken: {time.time() - start_time:.2f} seconds.")
        else:
            print("No rating method provided. Skipping rating.")

        # Visualize clustering
        print("Visualizing clustering...")
        start_time = time.time()
        self.visualization_method_1(self.embeddings, None, self.assigned_labels)
        print(f"Visualization completed. Time taken: {time.time() - start_time:.2f} seconds.")


        print("Visualizing actual categories...")
        start_time = time.time()
        if self.visualization_method_2 is not None:
            self.visualization_method_2(self.embeddings, self.categorizer, self.assigned_labels, self.true_single_labels)
        else:
            self.visualization_method_1(self.embeddings, self.categorizer, self.true_single_labels)
        print(f"Visualization of actual categories completed. Time taken: {time.time() - start_time:.2f} seconds.")