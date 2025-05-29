from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_data(file_path):
    """Load data from CSV file"""
    data = pd.read_csv(file_path)
    X = data.drop('TARGET', axis=1)
    y = data['TARGET']
    return X, y

def scale_numerical_features(X_train, X_val=None, features_to_scale=None):
    """Scale numerical features using StandardScaler"""
    if features_to_scale is None:
        return X_train, X_val, None

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])

    if X_val is not None:
        X_val_scaled = X_val.copy()
        X_val_scaled[features_to_scale] = scaler.transform(X_val[features_to_scale])
        return X_train_scaled, X_val_scaled, scaler

    return X_train_scaled, scaler
