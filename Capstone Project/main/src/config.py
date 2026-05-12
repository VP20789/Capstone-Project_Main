from pathlib import Path

# ============================================================
# Base Directory
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# Data Directories
# ============================================================

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# ============================================================
# Artifact Directories
# ============================================================

ARTIFACTS_DIR = BASE_DIR / "artifacts"

PLOTS_DIR = ARTIFACTS_DIR / "plots"

METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODELS_DIR = ARTIFACTS_DIR / "models"

# ============================================================
# Dataset Files
# ============================================================

DATA_FILE = RAW_DATA_DIR / "wdbc.data"

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR / "processed_data.csv"
)

MODEL_FILE = (
    MODELS_DIR / "rf_model.joblib"
)

# ============================================================
# ML Configuration
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70

VALIDATION_SIZE = 0.15

TEST_SIZE = 0.15

# ============================================================
# Random Forest Configuration
# ============================================================

N_ESTIMATORS = 200

# ============================================================
# Hyperparameter Tuning Grid
# ============================================================

PARAM_GRID = {

    'n_estimators': [200, 500, 800],

    'max_depth': [None, 5, 10, 20],

    'min_samples_split': [2, 5, 10],

    'min_samples_leaf': [1, 2, 4],

    'max_features': ['sqrt', 0.5, 1.0],

    'class_weight': [None, 'balanced']
}


# ============================================================
# GridSearchCV Configuration
# ============================================================

SCORING_METRIC = "roc_auc"

CV_FOLDS = 5

N_JOBS = -1

VERBOSE_LEVEL = 2


# ============================================================
# KMeans Configuration
# ============================================================

KMEANS_RANDOM_STATE = 42

KMEANS_N_INIT = 10

DEFAULT_N_CLUSTERS = 2

n_components=2

# ============================================================
# PCA Configuration
# ============================================================

PCA_COMPONENTS = 2