def dataset_shape(df):
    return df.shape


def missing_values(df):
    return df.isnull().sum().sum()


def duplicate_rows(df):
    return df.duplicated().sum()


def diagnosis_distribution(df):
    return df['Diagnosis'].value_counts()


def feature_count(df):
    return df.drop(columns=['ID', 'Diagnosis']).shape[1]