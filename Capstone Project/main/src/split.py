from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE
)


def stratified_split(X, y):

    temp_size = VALIDATION_SIZE + TEST_SIZE

    test_ratio = TEST_SIZE / temp_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )