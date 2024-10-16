import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

class CreditCardFraudDetector:
    def __init__(self, encoding_dim=14, threshold_percentile=99):
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.autoencoder = None
        self.encoder = None
        self.scaler_autoencoder = StandardScaler()
        self.threshold = None

    def build_model(self, input_dim):
        input_layer = keras.layers.Input(shape=(input_dim,))
        # Encoder
        encoded = keras.layers.Dense(input_dim, activation='tanh')(input_layer)
        encoded = keras.layers.Dense(self.encoding_dim * 2, activation='tanh')(encoded)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        # Decoder
        decoded = keras.layers.Dense(self.encoding_dim * 2, activation='tanh')(encoded)
        decoded = keras.layers.Dense(input_dim, activation='relu')(decoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)

        self.autoencoder = keras.models.Model(input_layer, decoded)
        self.encoder = keras.models.Model(input_layer, encoded)

        # Autoencoder
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, epochs=100, batch_size=256, validation_split=0.2):
        history = self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        # Computing the threshold for anomaly detection                         
        reconstructions = self.autoencoder.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, self.threshold_percentile)
        self.save_model_and_threshold()
        return history

    def save_model_and_threshold(self):
        # Create the directory for saving models if it doesn't exist
        model_dir = 'Models'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        joblib.dump(self.threshold, os.path.join(model_dir, 'threshold_value.joblib'))
        print(f"Threshold saved to {model_dir}/threshold_value.joblib")

        self.autoencoder.save(os.path.join(model_dir, 'autoencoder_model.h5'))
        print(f"Autoencoder model saved to {model_dir}/autoencoder_model.h5")

        joblib.dump(self.scaler_autoencoder, os.path.join(model_dir, 'scaler_autoencoder.pkl'))
        print(f"Scaler saved to {model_dir}/scaler.pkl")