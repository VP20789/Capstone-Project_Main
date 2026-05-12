# Breast Tumor Classification & Clustering Project

## Project Overview

This project focuses on the classification and clustering of breast tumor diagnostic data using Machine Learning techniques.

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset (WDBC)** to:

- classify tumors as:
  - Malignant (M)
  - Benign (B)

- perform unsupervised clustering analysis to identify natural groupings in tumor characteristics.

The project was structured using a modular Machine Learning pipeline approach with reusable source modules, configuration-driven workflows, and separated notebook phases.

---

# Problem Statement

The objective of this project is to build a Machine Learning pipeline capable of:

1. Predicting whether a tumor is malignant or benign.
2. Optimizing classification performance using Random Forest.
3. Evaluating model performance using medical-risk-focused metrics.
4. Performing clustering analysis to discover natural tumor groupings.
5. Maintaining reproducibility and modular project architecture.

---

# Dataset Information

Dataset:
- Breast Cancer Wisconsin Diagnostic Dataset (WDBC)

Files:
- `wdbc.data`
- `wdbc.names`

Dataset Characteristics:
- 569 observations
- 30 numerical features
- Binary classification target:
  - M = Malignant
  - B = Benign

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- Joblib
- Jupyter Notebook

---

# Project Folder Structure

```text
main/
│
├── data/
│   ├── raw/
│   │   ├── wdbc.data
│   │   └── wdbc.names
│   │
│   ├── interim/
│   │
│   └── processed/
│       └── processed_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_train_rf_classifier.ipynb
│   ├── 03b_hyperparameter_tuning.ipynb
│   ├── 04_evaluate.ipynb
│   ├── 04b_error_analysis.ipynb
│   └── 05_clustering.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── validate.py
│   ├── preprocess.py
│   ├── split.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   ├── thresholding.py
│   ├── error_analysis.py
│   └── clustering.py
│
├── artifacts/
│   ├── models/
│   │   └── rf_model.joblib
│   │
│   ├── metrics/
│   │   └── tuning_results.csv
│
├── requirements.txt
│
└── README.md

# Recommended Notebook Execution Order

Run the notebooks in the following sequence to reproduce the complete Machine Learning pipeline:

1. `01_eda.ipynb`
   - Perform Exploratory Data Analysis

2. `02_preprocess.ipynb`
   - Clean and preprocess the dataset

3. `03_train_rf_classifier.ipynb`
   - Train baseline Random Forest model

4. `03b_hyperparameter_tuning.ipynb`
   - Perform GridSearchCV hyperparameter tuning

5. `04_evaluate.ipynb`
   - Evaluate model performance and generate metrics

6. `04b_error_analysis.ipynb`
   - Analyze false positives, false negatives, and hard cases

7. `05_clustering.ipynb`
   - Perform KMeans and Hierarchical clustering analysis
