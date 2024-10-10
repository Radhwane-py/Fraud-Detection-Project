import tensorflow as tf
from tensorflow import keras
import numpy as np

class CreditCardFraudDetector:
    def __init__(self, encoding_dim=14, threshold_percentile=99):
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.autoencoder = None
        self.encoder = None
        self.scaler = None
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
        # computing the threshold for anomaly detection                         
        reconstructions = self.autoencoder.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, self.threshold_percentile)
        return history
