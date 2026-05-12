import pandas as pd


def encode_target(df):
    """
    Encode diagnosis column:
    M -> 1
    B -> 0
    """

    df['Diagnosis'] = df['Diagnosis'].map({
        'M': 1,
        'B': 0
    })

    return df


def drop_id_column(df):
    """
    Remove ID column.
    """

    return df.drop(columns=['ID'])


def split_features_target(df):
    """
    Split dataset into X and y.
    """

    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    return X, y