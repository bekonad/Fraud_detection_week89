%cd /content/Fraud_detection_week89
# Write updated script
import os
os.makedirs('/content/drive/MyDrive/scripts', exist_ok=True)
with open('/content/drive/MyDrive/scripts/fraud_detection_task3.py', 'w') as f:
    f.write('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
import shap
from datetime import datetime
from IPython.display import Image, display, HTML

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/content/drive/MyDrive/outputs/task3_output.log', mode='w'),
        logging.StreamHandler()
    ]
)

rng = np.random.default_rng(42)

def load_data_and_model():
    logging.debug("Loading data and model...")
    base_path = '/content/drive/MyDrive/Data/processed/'
    model_path = '/content/drive/MyDrive/reports/models/RandomForest_Creditcard.pkl'
    try:
        if not os.path.exists(base_path + 'creditcard_test_processed.csv'):
            logging.error(f"Missing file: {base_path + 'creditcard_test_processed.csv'}")
            raise FileNotFoundError(f"Missing file: {base_path + 'creditcard_test_processed.csv'}")
        if not os.path.exists(base_path + 'y_creditcard_test.csv'):
            logging.error(f"Missing file: {base_path + 'y_creditcard_test.csv'}")
            raise FileNotFoundError(f"Missing file: {base_path + 'y_creditcard_test.csv'}")
        X_creditcard_test = pd.read_csv(base_path + 'creditcard_test_processed.csv')
        y_creditcard_test = pd.read_csv(base_path + 'y_creditcard_test.csv').values.ravel()
        if os.path.exists(model_path):
            timestamp = datetime.fromtimestamp(os.path.getmtime(model_path))
            logging.debug(f"Model file timestamp: {timestamp} (EAT)")
            if timestamp < datetime(2025, 8, 20, 18, 0):
                logging.error(f"Model file is outdated (before 6:00 PM EAT, Aug 20, 2025). Please retrain with task2.py.")
                raise FileNotFoundError(f"Model file is outdated: {model_path}")
            if os.path.getsize(model_path) < 1000:
                logging.error(f"Model file is empty or too small: {model_path}")
                raise ValueError(f"Model file is empty or too small: {model_path}")
        else:
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        sample_indices = X_creditcard_test.sample(n=50, random_state=rng).index
        X_creditcard_test_sample = X_creditcard_test.loc[sample_indices].copy()
        y_creditcard_test_sample = y_creditcard_test[sample_indices]
        logging.debug(f"Sampled data: X_test shape={X_creditcard_test_sample.shape}, y_test shape={y_creditcard_test_sample.shape}")
        expected_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        actual_features = list(X_creditcard_test_sample.columns)
        if actual_features != expected_features:
            logging.error(f"Feature mismatch in X_test: Expected {expected_features}, got {actual_features}")
            raise ValueError(f"Feature mismatch in X_test: Expected {expected_features}, got {actual_features}")
        try:
            model_features = model.feature_names_in_
            logging.debug(f"Model features: {model_features.tolist()}")
            if len(model_features) != len(expected_features):
                logging.error(f"Model feature count mismatch: Expected {len(expected_features)} features, got {len(model_features)}")
                raise ValueError(f"Model feature count mismatch: Expected {len(expected_features)} features, got {len(model_features)}")
        except AttributeError:
            logging.warning("Model does not have feature_names_in_; assuming feature order matches training data")
        logging.debug(f"Final X_test features: {list(X_creditcard_test_sample.columns)}")
        return X_creditcard_test_sample, y_creditcard_test_sample, model
    except Exception as e:
        logging.error(f"Error loading data/model: {e}", exc_info=True)
        raise

def compute_shap_values(model, X_test):
    logging.debug("Computing SHAP values...")
    try:
        background = X_test.sample(n=25, random_state=rng)
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(X_test)
        logging.debug(f"Raw SHAP values type: {type(shap_values)}, shape: {np.array(shap_values).shape}")
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use class 1 (fraud) for binary classification
        elif not isinstance(shap_values, np.ndarray):
            logging.error(f"Unexpected SHAP values format: {type(shap_values)}")
            raise ValueError(f"Unexpected SHAP values format: {type(shap_values)}")
        logging.debug(f"Processed SHAP values shape: {shap_values.shape}, X_test shape: {X_test.shape}")
        if shap_values.shape[1] != X_test.shape[1]:
            logging.error(f"SHAP values shape mismatch: Expected {X_test.shape[1]} features, got {shap_values.shape[1]}")
            raise ValueError(f"SHAP values shape mismatch: Expected {X_test.shape[1]} features, got {shap_values.shape[1]}")
        return shap_values, explainer
    except Exception as e:
        logging.error(f"Error computing SHAP values: {e}", exc_info=True)
        raise

def plot_shap_summary(shap_values, X_test, dataset_name):
    logging.debug("Generating SHAP summary plot...")
    try:
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f'SHAP Summary Plot: {dataset_name}')
        plt.tight_layout()
        plot_path = f'/content/drive/MyDrive/reports/shap/summary_plot_{dataset_name}.png'
        os.makedirs('/content/drive/MyDrive/reports/shap', exist_ok=True)
        plt.savefig(plot_path)
        plt.close(fig)
        if os.path.exists(plot_path):
            logging.info(f"Saved SHAP summary plot: {plot_path}")
            display(Image(filename=plot_path))
        else:
            logging.error(f"Failed to save SHAP summary plot: {plot_path}")
            raise FileNotFoundError(f"Failed to save SHAP summary plot: {plot_path}")
    except Exception as e:
        logging.error(f"Error generating SHAP summary plot: {e}", exc_info=True)
        raise

def plot_shap_force(shap_values, explainer, X_test, dataset_name, instance_idx=0):
    logging.debug(f"Generating SHAP force plot for instance {instance_idx}...")
    try:
        base_value = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1]
        # Ensure shap_values is 2D for a single instance
        shap_values_instance = shap_values[[instance_idx]] if shap_values.ndim == 2 else shap_values
        logging.debug(f"Base value: {base_value}, SHAP values shape: {shap_values_instance.shape}, Features shape: {X_test.iloc[[instance_idx]].shape}")
        force_plot = shap.plots.force(base_value, shap_values_instance, X_test.iloc[[instance_idx]], feature_names=X_test.columns, matplotlib=False, show=False)
        plot_path_html = f'/content/drive/MyDrive/reports/shap/force_plot_{dataset_name}_instance_{instance_idx}.html'
        logging.debug(f"Attempting to save force plot to {plot_path_html}")
        try:
            os.makedirs(os.path.dirname(plot_path_html), exist_ok=True)
            shap.save_html(plot_path_html, force_plot)
            if os.path.exists(plot_path_html):
                logging.info(f"Saved SHAP force plot HTML: {plot_path_html}")
                with open(plot_path_html, 'r') as f:
                    logging.debug(f"HTML content preview: {f.read()[:200]}")
            else:
                logging.error(f"Failed to save SHAP force plot HTML: {plot_path_html}")
                raise FileNotFoundError(f"Failed to save SHAP force plot HTML: {plot_path_html}")
        except Exception as e:
            logging.error(f"Error saving SHAP force plot HTML: {e}", exc_info=True)
            raise
        return plot_path_html
    except Exception as e:
        logging.error(f"Error generating SHAP force plot: {e}", exc_info=True)
        raise

def interpret_shap(shap_values, X_test):
    logging.debug("Interpreting SHAP values...")
    try:
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'mean_shap_value': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_shap_value', ascending=False)
        insights = """SHAP Analysis Insights:
The SHAP summary plot shows the most important features driving fraud predictions.
Top 5 features:
{feature_importance_head}
Key drivers of fraud:
- Features like V4, V14, V10 (Creditcard dataset) have high SHAP values, indicating strong influence on fraud likelihood.
- Positive SHAP values push predictions towards fraud; negative values towards non-fraud.
The force plot illustrates how these features contribute to a specific transaction's fraud prediction.
""".format(feature_importance_head=feature_importance.head(5).to_string())
        logging.info(insights)
        output_path = '/content/drive/MyDrive/reports/shap/shap_insights.txt'
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(insights)
            if os.path.exists(output_path):
                logging.info(f"Saved SHAP insights to {output_path}")
                with open(output_path, 'r') as f:
                    logging.debug(f"SHAP insights content: {f.read()}")
            else:
                logging.error(f"Failed to save SHAP insights: {output_path}")
                raise FileNotFoundError(f"Failed to save SHAP insights: {output_path}")
        except Exception as e:
            logging.error(f"Error writing SHAP insights: {e}", exc_info=True)
            raise
    except Exception as e:
        logging.error(f"Error interpreting SHAP values: {e}", exc_info=True)
        raise

def main():
    try:
        X_creditcard_test, y_creditcard_test, model = load_data_and_model()
        shap_values, explainer = compute_shap_values(model, X_test=X_creditcard_test)
        plot_shap_summary(shap_values, X_test=X_creditcard_test, dataset_name="Creditcard")
        plot_shap_force(shap_values, explainer, X_test=X_creditcard_test, dataset_name="Creditcard", instance_idx=0)
        interpret_shap(shap_values, X_test=X_creditcard_test)
    except Exception as e:
        logging.error(f"Error occurred in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
''')

# Copy to local directory
!cp /content/drive/MyDrive/scripts/fraud_detection_task3.py /content/Fraud_detection_week89/scripts/fraud_detection_task3.py

# Run SHAP script
%run /content/Fraud_detection_week89/scripts/fraud_detection_task3.py

# Convert HTML to PNG
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def convert_html_to_png(html_path, output_png_path):
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        driver.get(f'file://{html_path}')
        time.sleep(5)  # Increased wait time
        driver.set_window_size(1200, 600)
        driver.save_screenshot(output_png_path)
        driver.quit()
        print(f"Saved PNG: {output_png_path}")
    except Exception as e:
        print(f"Error converting HTML to PNG: {e}")

try:
    convert_html_to_png(
        '/content/drive/MyDrive/reports/shap/force_plot_Creditcard_instance_0.html',
        '/content/drive/MyDrive/reports/shap/force_plot_Creditcard_instance_0.png'
    )
except Exception as e:
    print(f"Error converting HTML to PNG: {e}")

# Verify outputs
!ls -l /content/drive/MyDrive/reports/shap/
!cat /content/drive/MyDrive/outputs/task3_output.log