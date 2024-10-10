import pandas as pd
from scipy import stats
import numpy as np

def statistical_analysis(self, X_test, predicted_anomalies, feature_names):
    anomaly_data = X_test[predicted_anomalies]
    normal_data = X_test[~predicted_anomalies]

    results = []
    for i, feature in enumerate(feature_names):
        anomaly_values = self.scaler.inverse_transform(anomaly_data)[:, i]
        normal_values = self.scaler.inverse_transform(normal_data)[:, i]

        t_stat, p_value = stats.ttest_ind(anomaly_values, normal_values)

        results.append({
                'Feature': feature,
                'Anomaly Mean': np.mean(anomaly_values),
                'Normal Mean': np.mean(normal_values),
                'Difference': np.mean(anomaly_values) - np.mean(normal_values),
                'T-statistic': t_stat,
                'P-value': p_value
            })

    return pd.DataFrame(results)