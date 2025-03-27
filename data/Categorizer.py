from typing import List, Optional, Any

import numpy as np


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

    def get_labels_str(self, labels):
        """
        Get the string representation of a list of labels.

        Args:
            labels: List of labels.

        Returns:
            List of string representations of the labels.
        """
        return [self.get_label_str(label) for label in labels]


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

    def reduce_to_single_label(self, labels: List[List[int]]) -> Optional[np.ndarray[Any, np.dtype[Any]]]:
        """
        Reduce a list of lists of labels to a single list of labels. For every entry, select the label that occurs most often.

        Args:
            labels: List of lists of labels.

        Returns:
            List of labels.
        """
        label_frequency = {}
        for label_list in labels:
            if isinstance(label_list, int):
                return np.array(labels)
            for label in label_list:
                label_frequency[label] = label_frequency.get(label, 0) + 1
        single_labels = []
        for label_list in labels:
            single_label = max(label_list, key=lambda x: label_frequency[x])
            single_labels.append(single_label)
        return np.array(single_labels)

    def fit_lables(self, categories):
        """
        Fit the categorizer to the given labels.

        Args:
            categories: List of category labels (strings).

        Returns:
            None
        """
        numericalize_categories = []
        for label_str in categories:
            if self.use_parent_categories:
                label_str = label_str.split(".")[0]
            if label_str not in self.label_dict:
                self.label_dict[label_str] = self.unique_labels
                self.unique_labels += 1
            numericalize_categories.append(self.label_dict[label_str])
        return numericalize_categories