from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from src.config import (
    RANDOM_STATE,
    PARAM_GRID,
    SCORING_METRIC,
    CV_FOLDS,
    N_JOBS,
    VERBOSE_LEVEL
)


def perform_grid_search(X_train, y_train):

    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=model,

        param_grid=PARAM_GRID,

        scoring=SCORING_METRIC,

        cv=CV_FOLDS,

        n_jobs=N_JOBS,

        verbose=VERBOSE_LEVEL
    )

    grid_search.fit(X_train, y_train)

    return grid_search