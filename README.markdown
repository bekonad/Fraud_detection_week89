# Fraud Detection Project - 10 Academy KAIM Week 8 & 9

## Project Overview

This repository implements the 10 Academy KAIM Week 8 & 9 Challenge, developing a robust fraud detection system for e-commerce and banking transactions. The project processes datasets (`Fraud_Data.csv`, `creditcard.csv`, `IpAddress_to_Country.csv`), performs exploratory data analysis (EDA), engineers features, handles class imbalance with SMOTE, trains and evaluates three machine learning models (Logistic Regression, Random Forest, XGBoost), and provides optional SHAP analysis for interpretability. The workflow is organized into task-specific branches (`task1`, `task2`, `task3`) and a main branch (`main`), executed via Jupyter Notebooks in Google Colab (5.5 GiB memory, \~5-10 minutes runtime). Deliverables include processed datasets, trained models, evaluation metrics, visualizations, and reports (`blog_post.pdf`, `final_report.pdf`) stored in the `analysis/` directory.

## Objectives

- Develop an accurate fraud detection system for e-commerce and banking transactions.
- Address class imbalance using SMOTE.
- Train and evaluate Logistic Regression, Random Forest, and XGBoost models.
- Provide interpretable insights via SHAP analysis (optional).
- Deliver comprehensive reports and visualizations.
- Ensure reproducibility in Google Colab with a branch-based Git workflow.

## Repository Structure

```
Fraud_detection_week89/
│
├── notebooks/
│   ├── fraud_detection_task1.ipynb       # Data preprocessing and EDA
│   ├── fraud_detection_task2.ipynb       # Model training and evaluation
│   ├── fraud_detection_task3.ipynb       # SHAP analysis and report generation
│   ├── Fraud_detection_project.ipynb     # End-to-end pipeline
│
├── Data/
│   ├── Fraud_Data.csv                   # E-commerce transactions
│   ├── IpAddress_to_Country.csv         # IP-to-country mapping
│   ├── creditcard.csv                   # Banking transactions
│   ├── processed/                       # Processed datasets and preprocessors
│
├── scripts/
│   ├── fraud_detection_task1.py         # Task 1 script
│   ├── fraud_detection_task2.py         # Task 2 script
│   ├── fraud_detection_task3.py         # Task 3 script
│   ├── Fraud_detection_project.py       # Full pipeline script
│   ├── blog_post.tex                    # LaTeX for blog post
│   ├── final_report.tex                 # LaTeX for technical report
│
├── models/
│   ├── logistic_regression.pkl          # Logistic Regression model
│   ├── RandomForest_Creditcard.pkl      # Random Forest model
│   ├── xgboost_model.pkl                # XGBoost model
│
├── outputs/
│   ├── task1_output.log                 # Task 1 log
│   ├── task2_output.log                 # Task 2 log
│   ├── task3_output.log                 # Task 3 log
│   ├── project_output.log               # Full pipeline log
│   ├── predictions.csv                  # Model predictions
│   ├── evaluation_metrics.json          # Model performance metrics
│
├── reports/
│   ├── EDA_plots/
│   │   ├── class_distribution.png       # Class distribution
│   │   ├── purchase_patterns.png        # Purchase value/amount distributions
│   │   ├── bivariate_boxplots.png       # Feature-class relationships
│   ├── ConfusionMatrix.png              # Confusion matrices for models
│   ├── SHAP_summary.png                 # SHAP summary plot
│   ├── SHAP_force_plot.png              # SHAP force plot
│   ├── shap_insights.txt                # SHAP insights
│
├── analysis/
│   ├── blog_post.pdf                    # Medium-style blog post
│   ├── final_report.pdf                 # Technical report
│
├── .gitattributes                       # Git LFS configuration
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License
└── README.md
```

## Branches

- `main`: Contains the end-to-end pipeline (`Fraud_detection_project.ipynb`), documentation, and all deliverables.
- `task1`: Focuses on data preprocessing, EDA, and feature engineering.
- `task2`: Handles model training and evaluation.
- `task3`: Implements optional SHAP analysis and report generation.

## Notebooks

The project is driven by four Jupyter Notebooks in the `notebooks/` directory, designed for execution in Google Colab:

- `fraud_detection_task1.ipynb`:

  - **Purpose**: Loads, cleans, and preprocesses datasets; performs EDA and feature engineering.
  - **Tasks**:
    - Loads `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv`.
    - Cleans data (removes duplicates, handles missing values).
    - Maps IP addresses to countries using `IpAddress_to_Country.csv`.
    - Creates features (e.g., `time_to_purchase`).
    - Applies SMOTE for class imbalance.
    - Generates EDA visualizations (class distribution, purchase patterns, feature relationships).
  - **Outputs**:
    - Processed datasets in `Data/processed/` (e.g., `fraud_train_processed.csv`, `creditcard_train_processed.csv`).
    - Preprocessors in `Data/processed/` (e.g., `preprocessor.pkl`, `scaler.pkl`).
    - EDA plots in `reports/EDA_plots/`:
      - 
      - 
      - 
    - Log file: `outputs/task1_output.log`
  - **Branch**: `task1`

- `fraud_detection_task2.ipynb`:

  - **Purpose**: Trains and evaluates machine learning models.
  - **Tasks**:
    - Trains three models on the credit card dataset:
      - **Logistic Regression**: Simple, interpretable linear model.
      - **Random Forest**: Ensemble model for robust predictions.
      - **XGBoost**: Gradient-boosting model for high performance.
    - Evaluates models using AUC-PR and F1-Score.
    - Generates confusion matrices for visualization.
  - **Outputs**:
    - Models in `models/`:
      - `logistic_regression.pkl`
      - `RandomForest_Creditcard.pkl`
      - `xgboost_model.pkl`
    - Metrics in `outputs/evaluation_metrics.json`.
    - Predictions in `outputs/predictions.csv`.
    - Confusion matrix in `reports/ConfusionMatrix.png`:
      - 
    - Log file: `outputs/task2_output.log`
  - **Branch**: `task2`

- `fraud_detection_task3.ipynb`:

  - **Purpose**: Performs SHAP analysis for interpretability and generates reports.
  - **Tasks**:
    - Conducts SHAP analysis on the XGBoost model to identify key features (e.g., V4, V14, V10).
    - Generates SHAP summary and force plots.
    - Compiles LaTeX reports (`blog_post.pdf`, `final_report.pdf`) detailing model performance and insights.
  - **Outputs**:
    - SHAP outputs in `reports/`:
      - 
      - 
      - `shap_insights.txt`
    - Reports in `analysis/`:
      - `blog_post.pdf`: Medium-style narrative with model details and visualizations.
      - `final_report.pdf`: Technical report covering methodology, model performance, and SHAP analysis.
    - Log file: `outputs/task3_output.log`
  - **Branch**: `task3`

- `Fraud_detection_project.ipynb`:

  - **Purpose**: Executes the end-to-end pipeline combining Tasks 1–3.
  - **Tasks**: Includes data preprocessing, EDA, model training, evaluation, SHAP analysis, and report generation.
  - **Outputs**: All deliverables from Tasks 1–3.
  - **Log file**: `outputs/project_output.log`
  - **Branch**: `main`

All notebooks include `import logging` to prevent `NameError: name 'logging' is not defined`, with logs saved in `outputs/`.

## Analysis Directory

The `analysis/` directory contains the final reports generated by the project:

- `blog_post.pdf`:

  - A Medium-style blog post for a general audience, generated from `scripts/blog_post.tex`.
  - Describes the project’s goals, data exploration, and model training.
  - Highlights the three models (Logistic Regression, Random Forest, XGBoost) with performance metrics (e.g., XGBoost AUC-PR ≈ 0.89, F1-Score ≈ 0.88) and visualizations (e.g., `ConfusionMatrix.png`, `SHAP_summary.png`).
  - Includes embedded plots from `reports/EDA_plots/` and `reports/`.

- `final_report.pdf`:

  - A technical report for a professional audience, generated from `scripts/final_report.tex`.
  - Details the methodology, including data preprocessing, EDA, feature engineering, and model training.
  - Explicitly covers the three models, their configurations, performance metrics, and confusion matrices.
  - Includes SHAP analysis for XGBoost, highlighting key features (e.g., V4, V14, V10).
  - Embeds visualizations and references `outputs/evaluation_metrics.json`.

Both reports are compiled using `latexmk` and stored in `analysis/`, tracked with Git LFS due to their size.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/bekonad/Fraud_detection_week89.git
   cd Fraud_detection_week89
   git checkout main
   git lfs pull
   ```

2. **Set Up Google Colab**:

   - Open a Colab notebook.
   - Mount Google Drive:

     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Create directories:

     ```python
     import os
     folders = [
         '/content/drive/MyDrive/Data/processed',
         '/content/drive/MyDrive/reports/EDA_plots',
         '/content/drive/MyDrive/models',
         '/content/drive/MyDrive/outputs',
         '/content/drive/MyDrive/scripts',
         '/content/drive/MyDrive/notebooks',
         '/content/drive/MyDrive/analysis'
     ]
     for folder in folders:
         os.makedirs(folder, exist_ok=True)
     ```

3. **Install Dependencies**:

   - Create `requirements.txt`:

     ```bash
     cat <<EOT > requirements.txt
     pandas==2.2.2
     numpy==2.0.2
     matplotlib==3.10.0
     seaborn==0.13.2
     scikit-learn==1.6.1
     imbalanced-learn==0.14.0
     joblib==1.5.1
     shap==0.48.0
     selenium==4.25.0
     xgboost==2.1.1
     EOT
     ```
   - Install:

     ```bash
     pip install -r requirements.txt
     apt-get update && apt-get install -y chromium-chromedriver texlive-full latexmk git-lfs
     git lfs install
     ```

4. **Place Datasets**:

   - Place `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` in `Data/`.
   - Run `git lfs pull` to download large files if not already present.

5. **Run Notebooks or Scripts**:

   - Ensure `import logging` is included in all notebooks/scripts to avoid `NameError`.
   - Run in Colab:

     ```python
     %run notebooks/fraud_detection_task1.ipynb
     %run notebooks/fraud_detection_task2.ipynb
     %run notebooks/fraud_detection_task3.ipynb
     %run notebooks/Fraud_detection_project.ipynb
     ```
   - Or run scripts:

     ```bash
     python scripts/fraud_detection_task1.py
     python scripts/fraud_detection_task2.py
     python scripts/fraud_detection_task3.py
     python scripts/Fraud_detection_project.py
     ```

6. **Compile Reports**:

   ```bash
   latexmk -pdf scripts/blog_post.tex -outdir=analysis
   latexmk -pdf scripts/final_report.tex -outdir=analysis
   ```

7. **Push to GitHub**:

   - For each branch:

     ```bash
     git checkout task1
     git add notebooks/fraud_detection_task1.ipynb scripts/fraud_detection_task1.py Data/processed/* reports/EDA_plots/*
     git commit -m "Task 1: Data preprocessing and EDA"
     git push origin task1
     git lfs push --all origin task1
     ```

     Repeat for `task2`, `task3`, and `main`.
   - Merge to `main`:

     ```bash
     git checkout main
     git merge task1 task2 task3
     git push origin main
     git lfs push --all origin main
     ```

## Dataset Instructions

- **Raw Datasets**: `Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv` in `Data/`. Download via `git lfs pull` or from \[Google Drive Link\] (replace with actual link).
- **Processed Datasets**: Generated in `Data/processed/` by `fraud_detection_task1.ipynb` (e.g., `fraud_train_processed.csv`, `creditcard_train_processed.csv`).
- **Reports and Models**: Available in `analysis/` (`blog_post.pdf`, `final_report.pdf`) and `models/` (`logistic_regression.pkl`, `RandomForest_Creditcard.pkl`, `xgboost_model.pkl`). Download from \[Google Drive Link\] if not in repository.

## Task Details

### Task 1 (branch: `task1`)

- **Notebook**: `fraud_detection_task1.ipynb`
- **Script**: `fraud_detection_task1.py`
- Loads and preprocesses `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv`.
- Performs EDA, generating visualizations:
  - 
  - 
  - 
- Engineers features (e.g., `time_to_purchase`, IP-to-country mapping).
- Applies SMOTE for class imbalance.
- Saves processed datasets and preprocessors to `Data/processed/`.

### Task 2 (branch: `task2`)

- **Notebook**: `fraud_detection_task2.ipynb`
- **Script**: `fraud_detection_task2.py`
- Trains and evaluates three models:
  - **Logistic Regression**: Linear model for baseline performance.
  - **Random Forest**: Ensemble model for robust predictions.
  - **XGBoost**: Gradient-boosting model for optimal performance.
- Saves models to `models/` and metrics to `outputs/evaluation_metrics.json`.
- Generates confusion matrix:

  ![Confusion Matrix](reports/ConfusionMatrix.png).
- Approximate metrics:
  - XGBoost: AUC-PR ≈ 0.89, F1-Score ≈ 0.88
  - Random Forest: AUC-PR ≈ 0.8734, F1-Score ≈ 0.8743
  - Logistic Regression: AUC-PR ≈ 0.7453, F1-Score ≈ 0.7241

### Task 3 (branch: `task3`)

- **Notebook**: `fraud_detection_task3.ipynb`
- **Script**: `fraud_detection_task3.py`
- Performs SHAP analysis on the XGBoost model, generating:
  - 
  - 
  - `shap_insights.txt` (key features: V4, V14, V10)
- Compiles reports to `analysis/`:
  - `blog_post.pdf`: Includes model descriptions, metrics, and visualizations.
  - `final_report.pdf`: Details methodology, model performance, and SHAP analysis.

## Notes

- **Execution Environment**: Google Colab (5.5 GiB memory, \~5-10 minutes). For local execution, adjust paths and install dependencies.
- **Logging Fix**: Resolved `NameError: name 'logging' is not defined` by adding `import logging` in all notebooks/scripts.
- **Git LFS**: Tracks large files (CSV, PKL, PNG, PDF, HTML). Run `git lfs pull` after cloning.
- **Model Details in Reports**:
  - Both `blog_post.pdf` and `final_report.pdf` describe Logistic Regression, Random Forest, and XGBoost, including configurations, metrics, and visualizations (e.g., `ConfusionMatrix.png`, `SHAP_summary.png`).
  - Metrics are sourced from `outputs/evaluation_metrics.json`.
- **Troubleshooting**:
  - **Git Push Errors**: Configure Git LFS (`git lfs track "*.csv" "*.pkl" "*.png" "*.pdf" "*.html"`) and resolve conflicts (`git pull origin main --rebase`).
  - **LaTeX Errors**: Ensure `texlive-full` and `latexmk` are installed.
  - **SHAP Errors**: Verify `shap==0.48.0` and input shapes.
- **Repository**: https://github.com/bekonad/Fraud_detection_week89

## License

MIT License. See LICENSE for details.

## Contact

For issues, ontact \[bereketfeleke003@gmail.com or open a GitHub issue.