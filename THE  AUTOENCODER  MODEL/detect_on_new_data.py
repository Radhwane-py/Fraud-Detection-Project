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

def get_user_input():
    
    # Collect user inputs
    customer_age = int(input("Enter customer age: "))
    account_age = int(input("Enter account age (in years): "))
    transaction_amount = float(input("Enter transaction amount: "))
    merchant_category = float(input("Enter merchant category: "))
    transaction_type = input("Enter transaction type (e.g. cash_advance, transfer, purchase ): ")
    avg_transaction_amount = float(input("Enter average transaction amount: "))
    transaction_frequency = int(input("Enter transaction frequency: "))
    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))
    hour_of_day = int(input("Enter hour of the day (0-23): "))
    day_of_week = int(input("Enter day of the week (0=Monday, 6=Sunday): "))

    # Store the inputs in a dictionary format
    transaction = {
        'customer_age': customer_age,
        'account_age': account_age,
        'transaction_amount': transaction_amount,
        'merchant_category': merchant_category,
        'transaction_type': transaction_type,
        'avg_transaction_amount': avg_transaction_amount,
        'transaction_frequency': transaction_frequency,
        'latitude': latitude,
        'longitude': longitude,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week
    }

    return transaction

 # Main loop to collect transactions and make predictions
def main():
    # Collect transactions
    transactions = []
    while True:
        transaction = get_user_input()
        transactions.append(transaction)

        # If the user wants to enter another transaction
        another = input("Do you want to enter another transaction? (yes/no): ").strip().lower()
        if another != 'yes':
            break

    # Convert the input transactions to a DataFrame
    new_transactions_df = pd.DataFrame(transactions)

    # Make predictions
    predictions = predict(autoencoder, new_transactions_df, scaler_autoencoder, threshold)

    # Output the results
    print("\nPredictions for new transactions:")
    for i, (_, transaction) in enumerate(new_transactions_df.iterrows()):
        result = 'Fraudulent' if predictions[i] else 'Normal'
        print(f"Transaction {i+1}: {result}")

if __name__ == "__main__":
    main()