import pandas as pd


def get_false_positives(
    X_test,
    y_test,
    predictions,
    probabilities
):
    """
    Identify false positives.
    """

    fp_mask = (
        (y_test == 0) &
        (predictions == 1)
    )

    fp_df = X_test[fp_mask].copy()

    fp_df['actual'] = y_test[fp_mask]

    fp_df['predicted'] = predictions[fp_mask]

    fp_df['probability'] = probabilities[fp_mask]

    return fp_df


def get_false_negatives(
    X_test,
    y_test,
    predictions,
    probabilities
):
    """
    Identify false negatives.
    """

    fn_mask = (
        (y_test == 1) &
        (predictions == 0)
    )

    fn_df = X_test[fn_mask].copy()

    fn_df['actual'] = y_test[fn_mask]

    fn_df['predicted'] = predictions[fn_mask]

    fn_df['probability'] = probabilities[fn_mask]

    return fn_df