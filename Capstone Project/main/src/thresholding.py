import numpy as np


def apply_threshold(probabilities, threshold=0.5):
    """
    Convert probabilities into class predictions.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions