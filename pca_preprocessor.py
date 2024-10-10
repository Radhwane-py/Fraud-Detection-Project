import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump
import os

input_file = 'D:\Fraud Detection Project Main\Original data before PCA preprocessing\credit_card_transactions_training and testing.csv'
output_file = 'pca_results_for_training.csv'
n_components = 12

def load_data(file_path):
    
    df = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}")
    print(f"Shape of the dataset: {df.shape}")
    return df

def preprocess_with_pca(df, n_components):
    
    # Train or Predict
    if 'Class' in df.columns:
        print("Processing as training data (with 'Class').")
        X = df.drop(['Class'], axis=1)
        y = df['Class']
    else:
        print("Processing as prediction data (without 'Class').")
        X = df
        y = None

    # One hot encoding 
    X = pd.get_dummies(X, drop_first=True)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Data preprocessed with PCA. New shape: {X_pca.shape}")
    return X_pca, y, pca, scaler

def save_pca_results(X_pca, y, output_file):
    
    # Save PCA results to a CSV file.

    # Create a DataFrame with PCA results
    columns = [f'V{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=columns)

    if y is not None:
        df_pca['Class'] = y

    df_pca.to_csv(output_file, index=False)
    print(f"PCA results saved to {output_file}")

def main():
    # Load data
    df = load_data(input_file)

    # Preprocess with PCA
    X_pca, y, pca, scaler = preprocess_with_pca(df, n_components)

    # Save PCA results
    save_pca_results(X_pca, y, output_file)

    print("\nPCA Information:")
    print(f"Number of components: {n_components}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    model_dir = 'Models'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Save PCA and Scaler models
    dump(pca, os.path.join(model_dir, 'pca_model.joblib'))
    dump(scaler, os.path.join(model_dir, 'scaler_model.joblib'))
    print("\nPCA and Scaler models saved for future use.")

if __name__ == "__main__":
    main()
