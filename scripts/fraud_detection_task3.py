import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import joblib
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/task3_output.log'),
        logging.StreamHandler()
    ]
)

np.random.seed(42)

def load_processed_data():
    logging.info("Loading processed data...")
    base_path = 'data/processed/'
    try:
        X_fraud_train = pd.read_csv(base_path + 'fraud_train_processed.csv')
        X_fraud_test = pd.read_csv(base_path + 'fraud_test_processed.csv')
        X_creditcard_train = pd.read_csv(base_path + 'creditcard_train_processed.csv')
        X_creditcard_test = pd.read_csv(base_path + 'creditcard_test_processed.csv')
        y_fraud_train = pd.read_csv(base_path + 'y_fraud_train.csv').values.ravel()
        y_fraud_test = pd.read_csv(base_path + 'y_fraud_test.csv').values.ravel()
        y_creditcard_train = pd.read_csv(base_path + 'y_creditcard_train.csv').values.ravel()
        y_creditcard_test = pd.read_csv(base_path + 'y_creditcard_test.csv').values.ravel()
        logging.info("Data loaded successfully.")
        return (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test,
                X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test)
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    logging.info(f"Training {model_name} on {dataset_name}...")
    try:
        model.fit(X_train, y_train)
        os.makedirs('reports/models', exist_ok=True)
        joblib.dump(model, f'reports/models/{model_name}_{dataset_name}.pkl')
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
        os.makedirs('reports/confusion_matrices', exist_ok=True)
        plt.figure(figsize=(6, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Non-Fraud', 'Fraud']).plot(cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name} ({dataset_name})')
        plt.savefig(f'reports/confusion_matrices/{model_name}_{dataset_name}_cm.png')
        plt.close()
        logging.info(f"{model_name} ({dataset_name}) - AUC-PR: {auc_pr:.4f}, F1-Score: {f1:.4f}")
        logging.info(f"Classification Report for {model_name} ({dataset_name}):\n{report}")
        if model_name == "RandomForest":
            try:
                preprocessor = joblib.load('data/processed/preprocessor.pkl')
                feature_names = preprocessor.get_feature_names_out()
                feature_importances = pd.DataFrame({
                    'feature': feature_names[:X_train.shape[1]],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logging.info(f"Top 10 Feature Importances for {dataset_name}:\n{feature_importances.head(10)}")
            except Exception as e:
                logging.error(f"Error loading preprocessor: {e}")
                feature_importances = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logging.info(f"Top 10 Feature Importances for {dataset_name} (using column indices):\n{feature_importances.head(10)}")
        return auc_pr, f1, cm, report
    except Exception as e:
        logging.error(f"Error training/evaluating {model_name} on {dataset_name}: {e}")
        raise

def compare_models(results):
    logging.info("\nModel Comparison:")
    for model_name, dataset_name, auc_pr, f1, report in results:
        logging.info(f"{model_name} ({dataset_name}): AUC-PR = {auc_pr:.4f}, F1-Score = {f1:.4f}")
        logging.info(f"Classification Report:\n{report}")
    logging.info("\nModel Selection Justification:")
    logging.info("Logistic Regression is interpretable but may underperform on complex patterns.")
    logging.info("Random Forest is preferred for its ability to capture non-linear relationships and robustness to imbalanced data, especially on Creditcard dataset.")

def main():
    try:
        (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test,
         X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test) = load_processed_data()
        logging.info(f"X_fraud_train shape: {X_fraud_train.shape}, y_fraud_train shape: {y_fraud_train.shape}")
        logging.info(f"X_fraud_test shape: {X_fraud_test.shape}, y_fraud_test shape: {y_fraud_test.shape}")
        logging.info(f"X_creditcard_train shape: {X_creditcard_train.shape}, y_creditcard_train shape: {y_creditcard_train.shape}")
        logging.info(f"X_creditcard_test shape: {X_creditcard_test.shape}, y_creditcard_test shape: {y_creditcard_test.shape}")
        
        logreg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
        rf_fraud = GridSearchCV(rf_base, param_grid, cv=3, scoring='f1', n_jobs=-1)
        rf_credit = RandomForestClassifier(random_state=42, n_estimators=50, class_weight='balanced')
        
        results = []
        
        auc_pr_logreg_fraud, f1_logreg_fraud, cm_logreg_fraud, report_logreg_fraud = train_and_evaluate_model(
            logreg, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "LogisticRegression", "Fraud_Data")
        auc_pr_rf_fraud, f1_rf_fraud, cm_rf_fraud, report_rf_fraud = train_and_evaluate_model(
            rf_fraud, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "RandomForest", "Fraud_Data")
        auc_pr_logreg_credit, f1_logreg_credit, cm_logreg_credit, report_logreg_credit = train_and_evaluate_model(
            logreg, X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, "LogisticRegression", "Creditcard")
        auc_pr_rf_credit, f1_rf_credit, cm_rf_credit, report_rf_credit = train_and_evaluate_model(
            rf_credit, X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, "RandomForest", "Creditcard")
        
        results.extend([
            ("LogisticRegression", "Fraud_Data", auc_pr_logreg_fraud, f1_logreg_fraud, report_logreg_fraud),
            ("RandomForest", "Fraud_Data", auc_pr_rf_fraud, f1_rf_fraud, report_rf_fraud),
            ("LogisticRegression", "Creditcard", auc_pr_logreg_credit, f1_logreg_credit, report_logreg_credit),
            ("RandomForest", "Creditcard", auc_pr_rf_credit, f1_rf_credit, report_rf_credit)
        ])
        
        compare_models(results)
        
        os.makedirs('reports', exist_ok=True)
        with open('reports/model_results.txt', 'w') as f:
            f.write("Model Comparison Results:\n")
            for model_name, dataset_name, auc_pr, f1, report in results:
                f.write(f"{model_name} ({dataset_name}): AUC-PR = {auc_pr:.4f}, F1-Score = {f1:.4f}\n")
                f.write(f"Classification Report:\n{report}\n")
            f.write("\nModel Selection Justification:\n")
            f.write("Logistic Regression is interpretable but may underperform on complex patterns.\n")
            f.write("Random Forest is preferred for its ability to capture non-linear relationships and robustness to imbalanced data, especially on Creditcard dataset.")
        logging.info("Model results saved to reports/model_results.txt")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()