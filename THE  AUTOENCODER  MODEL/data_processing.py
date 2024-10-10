import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath, scaler):
    df = pd.read_csv(filepath)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns
