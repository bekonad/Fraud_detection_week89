import os
# Ensure scripts directory exists
os.makedirs('/content/drive/MyDrive/scripts', exist_ok=True)
os.makedirs('/content/drive/MyDrive/outputs', exist_ok=True)
os.makedirs('/content/drive/MyDrive/reports/models', exist_ok=True)
with open('/content/drive/MyDrive/scripts/fraud_detection_task2.py', 'w') as f:
    f.write
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import os
import logging
from IPython.display import Image, display

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/content/drive/MyDrive/outputs/task2_output.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Set random seed for reproducibility
np.random.seed(42)

def load_processed_data():
    """Load preprocessed datasets from Data/processed directory."""
    logging.info("Loading processed data...")
    base_path = '/content/drive/MyDrive/Data/processed/'
    required_files = [
        'fraud_train_processed.csv', 'fraud_test_processed.csv',
        'creditcard_train_processed.csv', 'creditcard_test_processed.csv',
        'y_fraud_train.csv', 'y_fraud_test.csv',
        'y_creditcard_train.csv', 'y_creditcard_test.csv'
    ]
    for file in required_files:
        if not os.path.exists(base_path + file):
            logging.error(f"Missing file: {base_path + file}")
            raise FileNotFoundError(f"Missing file: {base_path + file}")
    try:
        X_fraud_train = pd.read_csv(base_path + 'fraud_train_processed.csv')
        X_fraud_test = pd.read_csv(base_path + 'fraud_test_processed.csv')
        X_creditcard_train = pd.read_csv(base_path + 'creditcard_train_processed.csv')
        X_creditcard_test = pd.read_csv(base_path + 'creditcard_test_processed.csv')
        y_fraud_train = pd.read_csv(base_path + 'y_fraud_train.csv').values.ravel()
        y_fraud_test = pd.read_csv(base_path + 'y_fraud_test.csv').values.ravel()
        y_creditcard_train = pd.read_csv(base_path + 'y_creditcard_train.csv').values.ravel()
        y_creditcard_test = pd.read_csv(base_path + 'y_creditcard_test.csv').values.ravel()
        # Verify creditcard features
        expected_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        if list(X_creditcard_train.columns) != expected_features:
            logging.error(f"Creditcard train feature mismatch: Expected {expected_features}, got {list(X_creditcard_train.columns)}")
            raise ValueError("Creditcard train feature mismatch")
        if list(X_creditcard_test.columns) != expected_features:
            logging.error(f"Creditcard test feature mismatch: Expected {expected_features}, got {list(X_creditcard_test.columns)}")
            raise ValueError("Creditcard test feature mismatch")
        logging.info("Data loaded successfully.")
        logging.info(f"X_fraud_train shape: {X_fraud_train.shape}, y_fraud_train shape: {y_fraud_train.shape}")
        logging.info(f"X_fraud_test shape: {X_fraud_test.shape}, y_fraud_test shape: {y_fraud_test.shape}")
        logging.info(f"X_creditcard_train shape: {X_creditcard_train.shape}, y_creditcard_train shape: {y_creditcard_train.shape}")
        logging.info(f"X_creditcard_test shape: {X_creditcard_test.shape}, y_creditcard_test shape: {y_creditcard_test.shape}")
        return (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test,
                X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test)
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    """Train and evaluate a model, saving metrics and confusion matrix."""
    logging.info(f"Training {model_name} on {dataset_name}...")
    try:
        # Ensure model stores feature names
        model.fit(X_train, y_train)
        try:
            model.feature_names_in_ = X_train.columns
            logging.info(f"Stored feature names: {list(X_train.columns)}")
        except AttributeError:
            logging.warning("Model does not support feature_names_in_; feature order must match training data")
        # Force overwrite model file
        os.makedirs('/content/drive/MyDrive/reports/models', exist_ok=True)
        model_path = f'/content/drive/MyDrive/reports/models/{model_name}_{dataset_name}.pkl'
        if os.path.exists(model_path):
            os.remove(model_path)
            logging.info(f"Deleted existing model file: {model_path}")
        joblib.dump(model, model_path)
        # Verify model file
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            logging.error(f"Model file {model_path} is empty or too small!")
            raise ValueError(f"Model file {model_path} is empty or too small!")
        logging.info(f"Saved model: {model_path}, size: {os.path.getsize(model_path)} bytes")
        
        # Predict probabilities and labels
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
        
        # Save and display confusion matrix
        os.makedirs('/content/drive/MyDrive/reports/confusion_matrices', exist_ok=True)
        fig = plt.figure(figsize=(6, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Non-Fraud', 'Fraud']).plot(cmap='Blues', ax=fig.gca())
        plt.title(f'Confusion Matrix: {model_name} ({dataset_name})')
        plt.tight_layout()
        plot_path = f'/content/drive/MyDrive/reports/confusion_matrices/{model_name}_{dataset_name}_cm.png'
        plt.savefig(plot_path)
        plt.close(fig)
        logging.info(f"Saved confusion matrix: {plot_path}")
        display(Image(filename=plot_path))
        
        # Log metrics
        logging.info(f"{model_name} ({dataset_name}) - AUC-PR: {auc_pr:.4f}, F1-Score: {f1:.4f}")
        logging.info(f"Classification Report for {model_name} ({dataset_name}):\n{report}")
        
        # Feature importances for RandomForest
        if model_name == "RandomForest":
            feature_importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            logging.info(f"Top 10 Feature Importances for {dataset_name}:\n{feature_importances.head(10)}")
        
        return auc_pr, f1, cm, report
    except Exception as e:
        logging.error(f"Error training/evaluating {model_name} on {dataset_name}: {e}")
        raise

def compare_models(results):
    """Compare model performance across datasets."""
    logging.info("\nModel Comparison:")
    for model_name, dataset_name, auc_pr, f1, report in results:
        logging.info(f"{model_name} ({dataset_name}): AUC-PR = {auc_pr:.4f}, F1-Score = {f1:.4f}")
        logging.info(f"Classification Report:\n{report}")
    logging.info("\nModel Selection Justification:")
    logging.info("Logistic Regression is interpretable but may underperform on complex patterns.")
    logging.info("Random Forest is preferred for its ability to capture non-linear relationships and robustness to imbalanced data, especially on Creditcard dataset due to its high dimensionality and complex feature interactions.")

def main():
    """Main execution function for model training and evaluation."""
    try:
        # Load processed data
        (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test,
         X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test) = load_processed_data()

        # Initialize models
        logreg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        rf = RandomForestClassifier(random_state=42, n_estimators=5, class_weight='balanced', n_jobs=-1)
        results = []

        # Train and evaluate on Fraud_Data
        auc_pr_logreg_fraud, f1_logreg_fraud, cm_logreg_fraud, report_logreg_fraud = train_and_evaluate_model(
            logreg, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "LogisticRegression", "Fraud_Data")
        auc_pr_rf_fraud, f1_rf_fraud, cm_rf_fraud, report_rf_fraud = train_and_evaluate_model(
            rf, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "RandomForest", "Fraud_Data")

        # Train and evaluate on Creditcard Data
        auc_pr_logreg_credit, f1_logreg_credit, cm_logreg_credit, report_logreg_credit = train_and_evaluate_model(
            logreg, X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, "LogisticRegression", "Creditcard")
        auc_pr_rf_credit, f1_rf_credit, cm_rf_credit, report_rf_credit = train_and_evaluate_model(
            rf, X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, "RandomForest", "Creditcard")

        # Collect results
        results.extend([
            ("LogisticRegression", "Fraud_Data", auc_pr_logreg_fraud, f1_logreg_fraud, report_logreg_fraud),
            ("RandomForest", "Fraud_Data", auc_pr_rf_fraud, f1_rf_fraud, report_rf_fraud),
            ("LogisticRegression", "Creditcard", auc_pr_logreg_credit, f1_logreg_credit, report_logreg_credit),
            ("RandomForest", "Creditcard", auc_pr_rf_credit, f1_rf_credit, report_rf_credit)
        ])

        # Compare models
        compare_models(results)

        # Save results
        os.makedirs('/content/drive/MyDrive/reports', exist_ok=True)
        with open('/content/drive/MyDrive/reports/model_results.txt', 'w') as f:
            f.write("Model Comparison Results:\n")
            for model_name, dataset_name, auc_pr, f1, report in results:
                f.write(f"{model_name} ({dataset_name}): AUC-PR = {auc_pr:.4f}, F1-Score = {f1:.4f}\n")
                f.write(f"Classification Report:\n{report}\n")
            f.write("\nModel Selection Justification:\n")
            f.write("Logistic Regression is interpretable but may underperform on complex patterns.\n")
            f.write("Random Forest is preferred for its ability to capture non-linear relationships and robustness to imbalanced data, especially on Creditcard dataset due to its high dimensionality and complex feature interactions.")
        logging.info("Model results saved to /content/drive/MyDrive/reports/model_results.txt")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()