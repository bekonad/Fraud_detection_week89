# Fraud_detection_week89

Solution for 10 Academy KAIM Week 8&9 Challenge: Fraud Detection in E-commerce and Bank Transactions

## Overview

This project implements the 10 Academy Week 8&9 challenge, focusing on fraud detection in e-commerce and bank transactions. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, handling class imbalance, and model evaluation. The work is organized into separate branches for each task:

- `task1`: Data preprocessing, EDA, and feature engineering.
- `task2`: Model training, evaluation, and visualization.
- `task3` (optional): SHAP analysis for model interpretability.

## Setup

1. **Install dependencies**:
   - Ensure you have Python installed.
   - Run: `pip install -r requirements.txt` in your virtual environment.
2. **Place datasets**:
   - Download `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` from 10 Academy or Kaggle.
   - Place them in the `Data/raw/` directory.
3. **Activate virtual environment** (if using):
   - On Windows: `.\.venv\Scripts\Activate.ps1` in the terminal.
4. **Run the scripts**:
   - For Task 1: `scripts/fraud_detection_task1.py` (if in `scripts` folder).
   - For Task 2: `scripts/fraud_detection_task2.py`.
   - For Task 3: `scripts/fraud_detection_task2.py` (after adding SHAP code).

## Datasets

- **Sources**: Provided by 10 Academy or Kaggle (e.g., Credit Card Fraud Detection dataset).
- **Location**: Stored in `Data/raw/`; processed files are in `Data/processed/`.

## Project Overview

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
- Generates SHAP summary plots and saves them to `Data/processed/`.

## Branches

- `main`: Base branch with project setup and documentation.
- `task1`: Contains Task 1 solution.
- `task2`: Contains Task 2 solution.
- `task3`: Contains Task 3 solution (if implemented).

## Notes

- Large files (e.g., `creditcard.csv`) are ignored via `.gitignore` and handled with Git LFS if needed.
- Check GitHub[](https://github.com/bekonad/Fraud_detection_week89) for branch-specific changes.