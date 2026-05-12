from sklearn.ensemble import RandomForestClassifier

from src.config import (
    RANDOM_STATE,
    N_ESTIMATORS
)


def create_random_forest_model():
    """
    Create Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train Random Forest model.
    """

    model.fit(X_train, y_train)

    return model


def generate_predictions(model, X):
    """
    Generate predictions and probabilities.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities