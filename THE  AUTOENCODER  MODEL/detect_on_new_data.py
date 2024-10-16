import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load  

# Load the trained autoencoder model
autoencoder = load_model('Models/autoencoder_model.h5')

# Load the PCA, PCA_Scaler,autoencoder_Scaler and Threshold models
pca = load('Models/pca_model.joblib')
scaler_pca = load('Models/scaler_model_pca.joblib')
threshold = load('Models/threshold_value.joblib')  
scaler_autoencoder = load('Models/scaler_autoencoder.pkl')

def preprocess_with_pca(transactions, scaler_pca, pca):

    if isinstance(transactions, pd.DataFrame):
            transactions = transactions.values
    elif isinstance(transactions, list) or (isinstance(transactions, np.ndarray) and transactions.ndim == 1):
            transactions = np.array([transactions])

    # Preprocess the new data using the saved scaler and PCA model.
    
    # One-hot encoding
    transactions = pd.get_dummies(transactions, drop_first=True)

    # Standardize the data
    scaled_transactions = scaler_pca.transform(transactions)

    # Apply PCA transformation
    pca_transactions = pca.transform(scaled_transactions)

    return pca_transactions

def predict(detector, transactions,scaler_autoencoder, threshold):
    
    # Predict fraud using the trained model after preprocessing the transactions with PCA.
    
    # Preprocess the transactions with PCA and scaler
    preprocessed_transactions = preprocess_with_pca(transactions, scaler_pca, pca)

    auto_scaled_transactions = scaler_autoencoder.transform(preprocessed_transactions)

    # Use the loaded autoencoder model to make predictions
    reconstructions = detector.predict(auto_scaled_transactions)
    mse = np.mean(np.power(auto_scaled_transactions - reconstructions, 2), axis=1)

    # Use the threshold that was calculated during training
    return mse > threshold

# Load new transactions for prediction
new_transactions = pd.read_csv('D:\Fraud Detection Project Main\Original data before PCA preprocessing\credit_card_transactions_prediction.csv')  ##### Replace with the filepath

# Make predictions
predictions = predict(autoencoder, new_transactions,scaler_autoencoder, threshold)

# Output the results
print("\nPredictions for new transactions:")
for i, (_, transaction) in enumerate(new_transactions.iterrows()):
    result = 'Fraudulent' if predictions[i] else 'Normal'
    print(f"Transaction {i+1}: {result}")
