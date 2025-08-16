# Fraud_detection_week89

## Project Overview

This project implements the 10 Academy KAIM Week 8&9 Challenge: Fraud Detection in E-commerce and Bank Transactions. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, handling class imbalance, model training, evaluation, and optional SHAP analysis for interpretability. The work is organized into separate branches for each task.

## Branches

- `main`: Base branch with project setup and documentation.
- `task1`: Contains data preprocessing, EDA, and feature engineering.
- `task2`: Contains model training, evaluation, and visualization.
- `task3`: Contains optional SHAP analysis (if implemented).

## Setup

1. **Install Dependencies**:
   - Ensure Python is installed.
   - Run: `pip install -r requirements.txt` in your virtual environment.
2. **Activate Virtual Environment** (if using):
   - On Windows: `.\.venv\Scripts\Activate.ps1` in the terminal.
3. **Place Datasets**:
   - Download `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` from 10 Academy or Kaggle.
   - Place them in the `Data/raw/` directory.
4. **Run the Scripts**:
   - For Task 1: `python scripts/fraud_detection_task1.py` (if in `scripts/` folder).
   - For Task 2: `python scripts/fraud_detection_task2.py`.
   - For Task 3: `python scripts/fraud_detection_task3.py` (after adding SHAP code).

## Dataset Instructions

- **Raw and Processed Datasets**:
  - Stored via Git LFS in this repository (e.g., `Fraud_Data.csv`, `creditcard.csv`). After cloning, run `git lfs pull` to download them. Processed files are in `Data/processed/`.
- **Generated Reports and Models**:
  - Included in the `reports/` folder (if successfully pushed):
    - `reports/confusion_matrices/`: LogisticRegression_Creditcard_cm.png, LogisticRegression_Fraud_Data_cm.png, RandomForest_Creditcard_cm.png, RandomForest_Fraud_Data_cm.png
    - `reports/eda/`: bivariate_boxplots.png, univariate_distributions.png
    - `reports/models/`: LogisticRegression_Creditcard.pkl, LogisticRegression_Fraud_Data.pkl, RandomForest_Creditcard.pkl, RandomForest_Fraud_Data.pkl
    - `reports/`: model_results.txt
  - **Fallback**: If the `reports/` folder is not available, download from [Google Drive Link] and place them in the respective `reports/` subdirectories.

## Task Details

### Task 1 (branch: `task1`)

- Loads and preprocesses `Fraud_Data.csv` and `creditcard.csv`.
- Performs EDA (e.g., visualizations) and feature engineering.
- Handles class imbalance using techniques like oversampling.

### Task 2 (branch: `task2`)

- Trains and evaluates Logistic Regression and Random Forest models.
- Calculates performance metrics (AUC-PR, F1-Score).
- Generates confusion matrix visualizations and saves results to `reports/`.
- Justifies model selection based on performance.

### Task 3 (branch: `task3`, optional)

- Adds SHAP (SHapley Additive exPlanations) analysis for model interpretability.
- Generates SHAP summary plots and saves them to `reports/` (if implemented).

## Notes

- Large files (e.g., `creditcard.csv`) are ignored via `.gitignore` and handled with Git LFS.
- Check GitHub [https://github.com/bekonad/Fraud_detection_week89] for branch-specific changes.
- Sources: Datasets provided by 10 Academy or Kaggle (e.g., Credit Card Fraud Detection dataset).