import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def visualize_results(self, X_test, y_test, predicted_anomalies, mse):
    encoded_data = self.encoder.predict(X_test)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(encoded_data)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_test, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of transactions (actual frauds in yellow)')
    plt.savefig('tsne_visualization_actual.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=predicted_anomalies, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of transactions (predicted anomalies in yellow)')
    plt.savefig('tsne_visualization_predicted.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(data=pd.DataFrame({'MSE': mse, 'Fraud': y_test, 'Predicted Anomaly': predicted_anomalies}),
                 x='MSE', hue='Fraud', element='step', stat='density', common_norm=False)
    plt.title('Distribution of Reconstruction Errors')
    plt.savefig('reconstruction_error_distribution.png')
    plt.show()
