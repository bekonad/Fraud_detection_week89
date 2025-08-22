import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pandas import IntervalIndex
import joblib
import os
from IPython.display import Image, display

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    print("Loading datasets...")
    os.makedirs('/content/drive/MyDrive/Data/raw', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/Data/processed', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/reports/eda', exist_ok=True)

    try:
        fraud_data = pd.read_csv('/content/drive/MyDrive/Data/raw/Fraud_Data.csv')
        ip_to_country = pd.read_csv('/content/drive/MyDrive/Data/raw/IpAddress_to_Country.csv')
        creditcard_data = pd.read_csv('/content/drive/MyDrive/Data/raw/creditcard.csv')
        print("Data loaded successfully.")
        return fraud_data, ip_to_country, creditcard_data
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

def clean_data(fraud_data, creditcard_data):
    print("Cleaning data...")
    
    print("Fraud_Data Missing Values:\n", fraud_data.isnull().sum())
    fraud_data = fraud_data.dropna()  # Drop missing values
    fraud_data = fraud_data.drop_duplicates()
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

    print("Creditcard Missing Values:\n", creditcard_data.isnull().sum())
    creditcard_data = creditcard_data.dropna()
    creditcard_data = creditcard_data.drop_duplicates()

    return fraud_data, creditcard_data

def perform_eda(fraud_data, creditcard_data):
    print("Performing EDA...")
    
    # Class distribution analysis
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x='class', data=fraud_data)
    plt.title('Fraud_Data Class Distribution')
    plt.subplot(1, 2, 2)
    sns.countplot(x='Class', data=creditcard_data)
    plt.title('Creditcard Class Distribution')
    plt.tight_layout()
    plot_path = '/content/drive/MyDrive/reports/eda/class_distributions.png'
    plt.savefig(plot_path)
    plt.close(fig)
    print("Saved class distributions plot.")
    display(Image(filename=plot_path))

    # Univariate analysis
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(fraud_data['purchase_value'], bins=30)
    plt.title('Distribution of Purchase Value')
    plt.subplot(1, 2, 2)
    sns.histplot(creditcard_data['Amount'], bins=30)
    plt.title('Distribution of Transaction Amount')
    plt.tight_layout()
    plot_path = '/content/drive/MyDrive/reports/eda/univariate_distributions.png'
    plt.savefig(plot_path)
    plt.close(fig)
    print("Saved univariate distributions plot.")
    display(Image(filename=plot_path))

    # Bivariate analysis
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='class', y='purchase_value', data=fraud_data)
    plt.title('Purchase Value by Fraud Class')
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Class', y='Amount', data=creditcard_data)
    plt.title('Transaction Amount by Fraud Class')
    plt.tight_layout()
    plot_path = '/content/drive/MyDrive/reports/eda/bivariate_boxplots.png'
    plt.savefig(plot_path)
    plt.close(fig)
    print("Saved bivariate boxplots.")
    display(Image(filename=plot_path))

def merge_geolocation(fraud_data, ip_to_country):
    print("Merging geolocation data...")
    fraud_data['ip_address'] = pd.to_numeric(fraud_data['ip_address'], errors='coerce').fillna(0).astype(int)

    intervals = IntervalIndex.from_arrays(
        ip_to_country['lower_bound_ip_address'],
        ip_to_country['upper_bound_ip_address'],
        closed='both'
    )
    ip_to_country['interval'] = intervals

    def map_ip_to_country(ip):
        try:
            idx = intervals.get_indexer([ip])[0]
            return ip_to_country['country'].iloc[idx] if idx != -1 else 'Unknown'
        except:
            return 'Unknown'

    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)

    country_counts = fraud_data['country'].value_counts()
    threshold = 500
    rare_countries = country_counts[country_counts < threshold].index
    fraud_data['country'] = fraud_data['country'].apply(lambda x: 'Other' if x in rare_countries else x)
    print("Countries after grouping:\n", fraud_data['country'].value_counts())
    return fraud_data

def feature_engineering(fraud_data):
    print("Performing feature engineering...")
    
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600

    user_freq = fraud_data.groupby('user_id').size().reset_index(name='transaction_count')
    fraud_data = fraud_data.merge(user_freq, on='user_id', how='left')
    fraud_data['velocity'] = fraud_data['purchase_value'] / (fraud_data['time_since_signup'] + 1e-6)

    high_risk_countries = ['Nigeria', 'Ghana', 'Unknown', 'Other']
    fraud_data['ip_fraud_risk'] = fraud_data['country'].apply(lambda x: 1 if x in high_risk_countries else 0)
    print("IP fraud risk feature added:\n", fraud_data['ip_fraud_risk'].value_counts())
    return fraud_data

def handle_class_imbalance(X, y, sampling_strategy=0.5):
    print("Handling class imbalance...")
    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"Resampled shapes: X={X_resampled.shape}, y={y_resampled.shape}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error in SMOTE: {e}")
        raise

def transform_data(fraud_data, creditcard_data):
    print("Transforming data...")

    fraud_cat_cols = ['source', 'browser', 'sex', 'country']
    fraud_num_cols = ['purchase_value', 'age', 'hour_of_day', 'day_of_week',
                      'time_since_signup', 'transaction_count', 'velocity', 'ip_fraud_risk']
    creditcard_num_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), fraud_num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=100), fraud_cat_cols)
        ])

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
    fraud_feature_names = preprocessor.get_feature_names_out()

    scaler = StandardScaler()
    X_creditcard_train_transformed = scaler.fit_transform(X_creditcard_train)
    X_creditcard_test_transformed = scaler.transform(X_creditcard_test)
    creditcard_feature_names = X_creditcard.columns

    X_fraud_train_transformed = pd.DataFrame(X_fraud_train_transformed, columns=fraud_feature_names)
    X_fraud_test_transformed = pd.DataFrame(X_fraud_test_transformed, columns=fraud_feature_names)
    X_creditcard_train_transformed = pd.DataFrame(X_creditcard_train_transformed, columns=creditcard_feature_names)
    X_creditcard_test_transformed = pd.DataFrame(X_creditcard_test_transformed, columns=creditcard_feature_names)

    X_fraud_train_resampled, y_fraud_train_resampled = handle_class_imbalance(X_fraud_train_transformed, y_fraud_train)
    X_creditcard_train_resampled, y_creditcard_train_resampled = handle_class_imbalance(X_creditcard_train_transformed, y_creditcard_train)

    return (X_fraud_train_resampled, y_fraud_train_resampled, X_fraud_test_transformed, y_fraud_test,
            X_creditcard_train_resampled, y_creditcard_train_resampled, X_creditcard_test_transformed,
            y_creditcard_test, preprocessor, scaler)

def main():
    print("Starting main function...")
    try:
        fraud_data, ip_to_country, creditcard_data = load_data()
        fraud_data, creditcard_data = clean_data(fraud_data, creditcard_data)
        perform_eda(fraud_data, creditcard_data)
        fraud_data = merge_geolocation(fraud_data, ip_to_country)
        fraud_data = feature_engineering(fraud_data)
        (X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test,
         X_creditcard_train, y_creditcard_train, X_creditcard_test, y_creditcard_test,
         preprocessor, scaler) = transform_data(fraud_data, creditcard_data)

        output_dir = '/content/drive/MyDrive/Data/processed'
        os.makedirs(output_dir, exist_ok=True)
        output_files = {
            'fraud_train_processed.csv': X_fraud_train,
            'fraud_test_processed.csv': X_fraud_test,
            'creditcard_train_processed.csv': X_creditcard_train,
            'creditcard_test_processed.csv': X_creditcard_test,
            'y_fraud_train.csv': pd.Series(y_fraud_train),
            'y_fraud_test.csv': pd.Series(y_fraud_test),
            'y_creditcard_train.csv': pd.Series(y_creditcard_train),
            'y_creditcard_test.csv': pd.Series(y_creditcard_test)
        }

        for filename, df in output_files.items():
            file_path = os.path.join(output_dir, filename)
            df.to_csv(file_path, index=False)
            print(f"Successfully saved {file_path}")

        joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        print("Preprocessor and scaler saved.")
        print("Output files saved in Data/processed/")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()