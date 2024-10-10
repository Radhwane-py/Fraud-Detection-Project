import numpy as np
import pandas as pd

def detect_anomalies(self, data):
    reconstructions = self.autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    return mse > self.threshold, mse

def predict(self, transactions):
    if isinstance(transactions, pd.DataFrame):
        transactions = transactions.values
    elif isinstance(transactions, list) or (isinstance(transactions, np.ndarray) and transactions.ndim == 1):
        transactions = np.array([transactions])

    scaled_transactions = self.scaler.transform(transactions)
    reconstructions = self.autoencoder.predict(scaled_transactions)
    mse = np.mean(np.power(scaled_transactions - reconstructions, 2), axis=1)

    return mse > self.threshold
