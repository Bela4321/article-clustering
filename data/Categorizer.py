from typing import List



class Categorizer():
    def __init__(self):
        self.unique_labels = 0
        self.label_dict = {}
        self.use_parent_categories = False

    def fit_mulitlables(self, labels: List[List[str]]):
        """
        Fit the categorizer to the given labels.

        Args:
            labels: List of lists of category labels (strings).

        Returns:
            None
        """
        numericalize_categories = []
        for label_str_list in labels:
            numerical_label_list = []
            for label_str in label_str_list:
                if self.use_parent_categories:
                    label_str = label_str.split(".")[0]
                if label_str not in self.label_dict:
                    self.label_dict[label_str] = self.unique_labels
                    self.unique_labels += 1
                numerical_label_list.append(self.label_dict[label_str])
            numericalize_categories.append(numerical_label_list)
        return numericalize_categories

    def get_label_str(self, label):
        """
        Get the string representation of a label.

        Args:
            label: Label.

        Returns:
            String representation of the label.
        """
        for label_str, numerical_label in self.label_dict.items():
            if numerical_label == label:
                return label_str
        return "Unknown label"
