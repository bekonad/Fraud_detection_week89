import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load Data
def load_data():
    print("Loading datasets...", flush=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/eda', exist_ok=True)  # Changed from reports/eda

    fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_to_country = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    creditcard_data = pd.read_csv('data/raw/creditcard.csv')
    return fraud_data, ip_to_country, creditcard_data

# 2. Handle Missing Values and Data Cleaning
def clean_data(fraud_data, creditcard_data):
    # Clean fraud data
    print("Fraud_Data Missing Values:\n", fraud_data.isnull().sum(), flush=True)
    fraud_data = fraud_data.dropna()
    fraud_data = fraud_data.drop_duplicates()
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    # Clean creditcard data
    print("Creditcard Missing Values:\n", creditcard_data.isnull().sum(), flush=True)
    creditcard_data = creditcard_data.dropna()
    creditcard_data = creditcard_data.drop_duplicates()
    
    return fraud_data, creditcard_data

# 3. Exploratory Data Analysis (EDA)
def perform_eda(fraud_data, creditcard_data):
    print("Performing EDA...", flush=True)
    # Univariate distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(fraud_data['purchase_value'], bins=30)
    plt.title('Distribution of Purchase Value')
    plt.subplot(1, 2, 2)
    sns.histplot(creditcard_data['Amount'], bins=30)
    plt.title('Distribution of Transaction Amount')
    plt.tight_layout()
    plt.savefig('data/eda/univariate_distributions.png')
    plt.show()
    plt.close()
    
    # Bivariate analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='class', y='purchase_value', data=fraud_data)
    plt.title('Purchase Value by Fraud Class')
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Class', y='Amount', data=creditcard_data)
    plt.title('Transaction Amount by Fraud Class')
    plt.tight_layout()
    plt.savefig('data/eda/bivariate_boxplots.png')  # Fixed typo from bivariate_boxplots
    plt.show()
    plt.close()

# 4. Merge Datasets for Geolocation Analysis
def merge_geolocation(fraud_data, ip_to_country):
    print("Merging geolocation data...", flush=True)
    fraud_data['ip_address'] = pd.to_numeric(fraud_data['ip_address'], errors='coerce').fillna(0).astype(int)
    
    def map_ip_to_country(ip):
        try:
            country_row = ip_to_country[
                (ip_to_country['lower_bound_ip_address'] <= ip) & 
                (ip_to_country['upper_bound_ip_address'] >= ip)
            ]
            return country_row['country'].iloc[0] if not country_row.empty else 'Unknown'
        except:
            return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
    return fraud_data

# 5. Feature Engineering
def feature_engineering(fraud_data):
    print("Performing feature engineering...", flush=True)
    
    # Time-based features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - 
                                      fraud_data['signup_time']).dt.total_seconds() / 3600
    
    # Behavioral features
    user_freq = fraud_data.groupby('user_id').size().reset_index(name='transaction_count')
    fraud_data = fraud_data.merge(user_freq, on='user_id', how='left')
    fraud_data['velocity'] = fraud_data['purchase_value'] / (fraud_data['time_since_signup'] + 1e-6)
    
    return fraud_data

# 6. Handle Class Imbalance
def handle_class_imbalance(X, y, sampling_strategy=0.5):
    print("Handling class imbalance...", flush=True)
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# 7. Data Transformation
def transform_data(fraud_data, creditcard_data):
    print("Transforming data...", flush=True)

    # Define feature sets
    fraud_cat_cols = ['source', 'browser', 'sex', 'country']
    fraud_num_cols = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 
                      'time_since_signup', 'transaction_count', 'velocity']
    creditcard_num_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), fraud_num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), fraud_cat_cols)
        ])
    
    # Prepare data
    X_fraud = fraud_data[fraud_num_cols + fraud_cat_cols]
    y_fraud = fraud_data['class']
    X_creditcard = creditcard_data.drop('Class', axis=1)
    y_creditcard = creditcard_data['Class']
    
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)
    X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(
        X_creditcard, y_creditcard, test_size=0.2, random_state=42, stratify=y_creditcard)
    
    X_fraud_train_transformed = preprocessor.fit_transform(X_fraud_train)
    X_fraud_test_transformed = preprocessor.transform(X_fraud_test)
    
    scaler = StandardScaler()
    X_creditcard_train_transformed = scaler.fit_transform(X_creditcard_train)
    X_creditcard_test_transformed = scaler.transform(X_creditcard_test)
    
    X_fraud_train_resampled, y_fraud_train_resampled = handle_class_imbalance(
        X_fraud_train_transformed, y_fraud_train)
    X_creditcard_train_resampled, y_creditcard_train_resampled = handle_class_imbalance(
        X_creditcard_train_transformed, y_creditcard_train)
    
    return (X_fraud_train_resampled, y_fraud_train_resampled, X_fraud_test_transformed, y_fraud_test,
            X_creditcard_train_resampled, y_creditcard_train_resampled, X_creditcard_test_transformed, 
            y_creditcard_test, preprocessor, scaler)

# Main execution
def main():
    print("Starting main function...", flush=True)
    fraud_data, ip_to_country, creditcard_data = load_data()
    print("Data loaded successfully.", flush=True)
    fraud_data, creditcard_data = clean_data(fraud_data, creditcard_data)
    print("Data cleaned.", flush=True)
    perform_eda(fraud_data, creditcard_data)
    print("EDA completed, plots saved.", flush=True)
    fraud_data = merge_geolocation(fraud_data, ip_to_country)
    print("Geolocation merged.", flush=True)
    fraud_data = feature_engineering(fraud_data)
    print("Feature engineering completed.", flush=True)
    (X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test,
     X_creditcard_train, y_creditcard_train, X_creditcard_test, y_creditcard_test,
     preprocessor, scaler) = transform_data(fraud_data, creditcard_data)
    print("Data transformation completed.", flush=True)
    
    # Save data
    pd.DataFrame(X_fraud_train).to_csv('data/processed/fraud_train_processed.csv', index=False)
    pd.DataFrame(X_fraud_test).to_csv('data/processed/fraud_test_processed.csv', index=False)
    pd.DataFrame(X_creditcard_train).to_csv('data/processed/creditcard_train_processed.csv', index=False)
    pd.DataFrame(X_creditcard_test).to_csv('data/processed/creditcard_test_processed.csv', index=False)
    pd.DataFrame(y_fraud_train).to_csv('data/processed/y_fraud_train.csv', index=False)
    pd.DataFrame(y_fraud_test).to_csv('data/processed/y_fraud_test.csv', index=False)
    pd.DataFrame(y_creditcard_train).to_csv('data/processed/y_creditcard_train.csv', index=False)
    pd.DataFrame(y_creditcard_test).to_csv('data/processed/y_creditcard_test.csv', index=False)
    print("Output files saved in /data/processed/", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}", flush=True)