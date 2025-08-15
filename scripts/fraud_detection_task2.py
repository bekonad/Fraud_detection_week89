import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import joblib
import os

np.random.seed(42)

def load_processed_data():
    print("Loading processed data...", flush=True)
    base_path = 'data/processed/'
    X_fraud_train = pd.read_csv(base_path + 'fraud_train_processed.csv')
    X_fraud_test = pd.read_csv(base_path + 'fraud_test_processed.csv')
    X_creditcard_train = pd.read_csv(base_path + 'creditcard_train_processed.csv')
    X_creditcard_test = pd.read_csv(base_path + 'creditcard_test_processed.csv')
    y_fraud_train = pd.read_csv(base_path + 'y_fraud_train.csv').values.ravel()
    y_fraud_test = pd.read_csv(base_path + 'y_fraud_test.csv').values.ravel()
    y_creditcard_train = pd.read_csv(base_path + 'y_creditcard_train.csv').values.ravel()
    y_creditcard_test = pd.read_csv(base_path + 'y_creditcard_test.csv').values.ravel()
    print("Data loaded successfully.", flush=True)
    return (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test,
            X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    print(f"Training {model_name} on {dataset_name}...", flush=True)
    model.fit(X_train, y_train)
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(model, f'data/models/{model_name}_{dataset_name}.pkl')
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs('data/confusion_matrices', exist_ok=True)
    plt.figure(figsize=(6, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Non-Fraud', 'Fraud']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} ({dataset_name})')
    plt.savefig(f'data/confusion_matrices/{model_name}_{dataset_name}_cm.png')
    plt.show()
    plt.close()
    print(f"{model_name} ({dataset_name}) - AUC-PR: {auc_pr:.4f}, F1-Score: {f1:.4f}", flush=True)
    return auc_pr, f1, cm

def compare_models(results):
    print("\nModel Comparison:", flush=True)
    for model_name, dataset_name, auc_pr, f1 in results:
        print(f"{model_name} ({dataset_name}): AUC-PR = {auc_pr:.4f}, F1-Score = {f1:.4f}", flush=True)
    print("\nModel Selection Justification:", flush=True)
    print("Logistic Regression is interpretable but may underperform on complex patterns.")
    print("Random Forest is chosen as the best model if it shows higher AUC-PR and F1-Score, as it captures non-linear relationships and is robust to imbalanced data.")

def main():
    (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test,
     X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test) = load_processed_data()
    print(f"X_fraud_train shape: {X_fraud_train.shape}, y_fraud_train shape: {y_fraud_train.shape}")
    print(f"X_fraud_test shape: {X_fraud_test.shape}, y_fraud_test shape: {y_fraud_test.shape}")
    print(f"X_creditcard_train shape: {X_creditcard_train.shape}, y_creditcard_train shape: {y_creditcard_train.shape}")
    print(f"X_creditcard_test shape: {X_creditcard_test.shape}, y_creditcard_test shape: {y_creditcard_test.shape}")
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    results = []
    auc_pr_logreg_fraud, f1_logreg_fraud, cm_logreg_fraud = train_and_evaluate_model(
        logreg, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "LogisticRegression", "Fraud_Data")
    auc_pr_rf_fraud, f1_rf_fraud, cm_rf_fraud = train_and_evaluate_model(
        rf, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "RandomForest", "Fraud_Data")
    auc_pr_logreg_credit, f1_logreg_credit, cm_logreg_credit = train_and_evaluate_model(
        logreg, X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, "LogisticRegression", "Creditcard")
    auc_pr_rf_credit, f1_rf_credit, cm_rf_credit = train_and_evaluate_model(
        rf, X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, "RandomForest", "Creditcard")
    results.extend([
        ("LogisticRegression", "Fraud_Data", auc_pr_logreg_fraud, f1_logreg_fraud),
        ("RandomForest", "Fraud_Data", auc_pr_rf_fraud, f1_rf_fraud),
        ("LogisticRegression", "Creditcard", auc_pr_logreg_credit, f1_logreg_credit),
        ("RandomForest", "Creditcard", auc_pr_rf_credit, f1_rf_credit)
    ])
    compare_models(results)
    os.makedirs('data', exist_ok=True)
    with open('data/model_results.txt', 'w') as f:
        f.write("Model Comparison Results:\n")
        for model_name, dataset_name, auc_pr, f1 in results:
            f.write(f"{model_name} ({dataset_name}): AUC-PR = {auc_pr:.4f}, F1-Score = {f1:.4f}\n")
        f.write("\nModel Selection Justification:\n")
        f.write("Logistic Regression is interpretable but may underperform on complex patterns.\n")
        f.write("Random Forest is chosen as the best model if it shows higher AUC-PR and F1-Score, as it captures non-linear relationships and is robust to imbalanced data.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}", flush=True)