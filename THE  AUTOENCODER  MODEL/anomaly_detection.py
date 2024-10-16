import numpy as np

def detect_anomalies(self, data):
    reconstructions = self.autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    return mse > self.threshold, mse
