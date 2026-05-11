# Capstone-Project_Main
Breast Tumor Risk Identification &amp; Diagnostic Pattern Segmentation Using Random Forest Analytics
Using Random Forest Analytics on the UCI WDBC Dataset

Project Charter: Breast Tumor Classification with Random Forest + Pattern Segmentation (UCI WDBC)
Project Overview
This project will develop a reproducible machine learning workflow to classify breast tumors as Malignant (M) or Benign (B) using the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository. The project also includes clustering to identify feature-based diagnostic pattern groups and summarize them as cluster profiles for interpretation and communication. This work is intended for learning and demonstration only and must not be presented as medical advice or a clinical diagnostic tool.
Problem Statement
The project objective is to build a single Random Forest Classifier that predicts whether a tumor is malignant or benign from the provided diagnostic features. The workflow must demonstrate professional machine learning practice through data validation, preprocessing, stratified train/validation/test splitting, hyperparameter tuning, threshold justification, rigorous evaluation, and reproducible packaging rather than comparing multiple supervised algorithms.
Scope
In Scope
•	Random Forest Classifier as the only supervised model.
•	Target classification of diagnosis label M versus B.
•	Data validation, preprocessing, and reproducible train/validation/test workflow.
•	Hyperparameter tuning using GridSearch or Randomized Search with the required starting grid.
•	Evaluation using confusion matrix, ROC curve, ROC-AUC, precision, recall, F1-score, and threshold analysis.
•	Clustering using scaled features with both K-means and hierarchical clustering, including cluster profiles.
•	Model artifact creation and inference script for prediction output.
Out of Scope
•	Multi-model comparison write-up or use of supervised algorithms other than Random Forest Classifier.
•	Tuning or repeated decision-making on the test set.
•	Clinical deployment or any claim of diagnostic capability.
Dataset and Target
The dataset contains 569 observations, 30 real-valued diagnostic features, an ID column that exists but is not used for training, and no missing values. The target variable is the diagnosis label, with M representing malignant tumors and B representing benign tumors. For modeling and metric calculation, malignant should be treated as the positive class so recall can directly reflect sensitivity to missed malignant cases, which aligns with the project guidance to protect recall for malignant tumors.
Success Metrics
Model selection should prioritize ROC-AUC while protecting Recall for malignant cases, because the brief explicitly states that the chosen model should perform strongly on ROC-AUC while protecting Recall, especially for malignant tumors. F1-score should be used as a balancing metric during threshold evaluation, with precision, confusion matrix, and ROC curve serving as supporting diagnostics.
Recommended metric hierarchy:
•	ROC-AUC as the primary ranking metric.
•	Recall for class M as the primary operational safeguard metric.
•	F1-score for class M as the threshold-balancing metric.
•	Precision, confusion matrix, and ROC curve as supporting evaluation views.
Split Strategy and Reproducibility
The project must use a stratified train/validation/test split with a fixed random state, and the test set must be used once for final reporting only. A practical implementation is a 70 percent training split, 15 percent validation split, and 15 percent test split, while preserving the class distribution across all subsets. Leakage prevention rules should ensure that all preprocessing decisions, tuning, and threshold selection are driven by training and validation data only.
Deliverables
The code deliverables must include the following components:
Below is the placeholder.
•	01_eda.ipynb
•	02_preprocess.ipynb or src/preprocess.py
•	03_train_rf_classifier.ipynb or src/train.py
•	04_evaluate.ipynb
•	05_clustering.ipynb
•	rf_model.joblib or equivalent model artifact
•	inference.py that prints predicted class, malignancy probability or class probabilities, and threshold used
The final report must cover background, dataset summary, validation checks, preprocessing, split strategy, Random Forest training and tuning, final evaluation, threshold decision, error analysis, clustering results, recommendations, limitations, and next steps.
Team Roles
The problem statement defines a five-member team structure with role ownership across project integration, data engineering, Random Forest modeling, tuning and evaluation, and clustering insights. This role split supports clear accountability for validation, preprocessing, model training, threshold selection, error analysis, clustering workflow, and final communication.
Governance Decisions
The project will encode the target so that M is the positive class and B is the negative class for evaluation consistency. The Random Forest tuning process will start from the required hyperparameter grid: n_estimator of 200, 500, and 800; max_depth of None, 5, 10, and 20; min_samples_split of 2, 5, and 10; min_samples_leaf of 1, 2, and 4; max_features of sqrt, 0.5, and 1.0; and class_weight of None or balanced, all under a fixed documented random state.
Proposed Repository Structure
breast-tumor-rf-clustering/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_train_rf_classifier.ipynb
│   ├── 04_evaluate.ipynb
│   └── 05_clustering.ipynb
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
│   ├── clustering.py
│   └── profiles.py
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── plots/
├── reports/
│   └── final_report.md
├── tests/
└── inference.py

This structure supports the required deliverables while separating exploratory notebooks from reusable source code and persisted outputs.
Timeline Summary
1.	Validate dataset and confirm schema, target, and exclusions.
2.	Build preprocessing and stratified split pipeline.
3.	Train and tune the Random Forest classifier using the required search space.
4.	Evaluate on validation data, justify the threshold, and lock the final model.
5.	Run final one-time test evaluation and generate plots and metrics
6.	Perform K-means and hierarchical clustering on scaled features and prepare cluster profiles
7.	Package artifacts, inference script, report, and presentation-ready outputs.
Risks and Controls
The main project risks are data leakage, over-reliance on accuracy, misuse of the test set, and weak interpretation of false negatives for malignant tumors. These risks are controlled by stratified splitting, fixed seeds, one-time test use, threshold justification, and evaluation centered on ROC-AUC, Recall, F1, and confusion-matrix analysis rather than accuracy alone.
