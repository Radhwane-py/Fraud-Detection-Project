from model import CreditCardFraudDetector
from data_processing import load_and_preprocess_data
from evaluation import evaluate_performance
from anomaly_detection import detect_anomalies 
from visualization import visualize_results
from utils_stats import statistical_analysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Initialize the detector
    detector = CreditCardFraudDetector()

    # Load and preprocess data
                    #######
    filepath = 'D:\Fraud Detection Project Main\pca_results_for_training.csv'  ##### the path to the file 
                    #######
    
    X, y, feature_names = load_and_preprocess_data(filepath, detector.scaler)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build and train model
    detector.build_model(X_train.shape[1])
    history = detector.train(X_train)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

    # Detect anomalies
    predicted_anomalies, mse = detect_anomalies(X_test)

    # Visualize results
    visualize_results(X_test, y_test, predicted_anomalies, mse)

    # Statistical analysis
    stats_df = statistical_analysis(X_test, predicted_anomalies, feature_names)

    # Print results
    print(f"\nNumber of transactions in test set: {len(X_test)}")
    print(f"Number of actual frauds in test set: {np.sum(y_test)}")
    print(f"Number of predicted anomalies: {np.sum(predicted_anomalies)}")
    print(f"Predicted anomaly rate: {np.mean(predicted_anomalies):.2%}")

    print("\nTop 5 most significant features:")
    print(stats_df.sort_values('P-value').head())

    # Save the statistical analysis to CSV
    stats_df.to_csv('statistical_analysis.csv', index=False)

    print("\nFull statistical analysis saved to 'statistical_analysis.csv'")
    print("Visualizations saved as PNG files in the current directory")

    # Evaluate model performance
    evaluate_performance(y_test, predicted_anomalies)

if __name__ == "__main__":
    main()
