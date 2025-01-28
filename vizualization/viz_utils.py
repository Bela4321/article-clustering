from typing import Optional, Any

import numpy as np
from fontTools.qu2cu.qu2cu import List
from numpy import ndarray, dtype


def reduce_to_single_label(labels: List[List[int]]) -> Optional[ndarray[Any, dtype[Any]]]:
    """
    Reduce a list of lists of labels to a single list of labels. For every entry, select the label that occurs most often.

    Args:
        labels: List of lists of labels.

    Returns:
        List of labels.
    """
    label_frequency = {}
    for label_list in labels:
        for label in label_list:
            label_frequency[label] = label_frequency.get(label, 0) + 1
    single_labels = []
    for label_list in labels:
        single_label = max(label_list, key=lambda x: label_frequency[x])
        single_labels.append(single_label)
    return np.array(single_labels)
