import time

from data.Categorizer import Categorizer
from vizualization.viz_umap import plot_in_2d as umap_plot

class SequentialClustering:
    def __init__(self, data_loader, embedding_method, clustering_method, rating_method= None, visualization_method = umap_plot):
        """
        Initializes the SequentialClustering pipeline with dependency injection.

        Parameters:
            data_loader (function): Function to load data.
            embedding_method (function): Function to create embeddings from data.
            clustering_method (function): Function to perform clustering.
            rating_method (function): Function to rate clustering performance.
            visualization_method (function): Function to visualize clusters.
        """
        self.data_loader = data_loader
        self.embedding_method = embedding_method
        self.clustering_method = clustering_method
        self.rating_method = rating_method
        self.visualization_method = visualization_method

        self.categorizer = Categorizer()
        # store intermediate results
        self.data = None
        self.labels = None
        self.embeddings = None
        self.cluster_labels = None

    def run(self, base_query, additional_query=None, load_limit=None):
        """
        Executes the sequential clustering pipeline.

        Parameters:
            base_query (str): Base query string to filter or load specific data (e.g., 'quantum').
            reduce_labels (function, optional): Function to reduce multi-labels to single labels for visualization.
            additional_query (str, optional): Additional query parameter for data filtering. Default is None.

        Returns:
            None
            :param base_query:
            :param additional_query:
            :param load_limit:
        """
        # Construct the full query
        full_query = f"abstract.str.contains('{base_query}')"
        if additional_query:
            full_query += f" and {additional_query}"

        # Load data
        print("Loading data...")
        start_time = time.time()
        if load_limit is None:
            self.data, self.labels = self.data_loader(self.categorizer, full_query)
        else:
            self.data, self.labels = self.data_loader(self.categorizer, full_query, load_limit)
        print(f"Data loaded. {len(self.data)} entries. Time taken: {time.time() - start_time:.2f} seconds.")

        # Create embeddings
        print("Creating embeddings...")
        start_time = time.time()
        self.embeddings, self.cluster_labels = self.embedding_method(self.data)
        print(f"Embeddings created with shape {self.embeddings.shape}. Time taken: {time.time() - start_time:.2f} seconds.")

        if self.cluster_labels is None:
            # Perform clustering
            print("Performing clustering...")
            start_time = time.time()
            self.cluster_labels = self.clustering_method(self.embeddings)
            print(f"Clustering done. Time taken: {time.time() - start_time:.2f} seconds.")
        else:
            print("Clustering labels provided. Skipping clustering step.")

        if self.rating_method is not None:
            # Rate clustering
            print("Rating clustering...")
            start_time = time.time()
            purity = self.rating_method(self.cluster_labels, self.labels)
            print(f"Purity: {purity}. Time taken: {time.time() - start_time:.2f} seconds.")
        else:
            print("No rating method provided. Skipping rating.")

        # Visualize clustering
        print("Visualizing clustering...")
        start_time = time.time()
        self.visualization_method(self.embeddings, None,self.cluster_labels)
        print(f"Visualization completed. Time taken: {time.time() - start_time:.2f} seconds.")


        print("Visualizing actual categories...")
        start_time = time.time()
        self.visualization_method(self.embeddings, self.categorizer,self.labels, needs_reduction=True)
        print(f"Visualization of actual categories completed. Time taken: {time.time() - start_time:.2f} seconds.")