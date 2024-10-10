# Credit Card Fraud Detection using Autoencoder with PCA

This project implements an anomaly detection system to identify fraudulent credit card transactions. It utilizes an autoencoder neural network to learn the patterns of normal transactions and detect anomalies, which could potentially be fraudulent. The dataset used has been preprocessed using Principal Component Analysis (PCA) to change the dimensionality in order to protect users' identities and sensitive features and improve performance.

## Project Overview
The project leverages an autoencoder model to detect credit card fraud by learning to reconstruct normal transactions. The anomaly detection process relies on the reconstruction error; transactions that deviate significantly from the learned pattern of normal transactions are flagged as anomalies.

## Key Features:
Autoencoder for unsupervised anomaly detection.
Data preprocessing with PCA to change dimensionality and reduce noise.
Visualizations using t-SNE to observe how normal and fraudulent transactions differ.
Statistical analysis on detected anomalies versus normal transactions.

## Data Preprocessing
The dataset used in this project has already been preprocessed with Principal Component Analysis **(PCA)**.
PCA generally reduces the dimensionality of the dataset by transforming the original features into a set of uncorrelated features while retaining as much variance as possible. This preprocessing step enhances the performance of the model by reducing noise and making it easier to detect anomalies.

The Class column in the dataset is used for evaluation:

**0: Normal transaction.**
**1: Fraudulent transaction.**

## Model Details
The autoencoder model is an unsupervised neural network designed to learn an efficient representation of normal transactions. It consists of:

**Encoder**: Compresses the input data (transactions) into a lower-dimensional representation.
**Decoder**: Reconstructs the original input from the compressed representation.
The model is trained to minimize the reconstruction error on normal transactions. Fraudulent transactions, which differ significantly from normal transactions, will have higher reconstruction errors and will be flagged as anomalies.

## Model Configuration:
encoding_dim: The size of the compressed layer.
threshold_percentile: Percentile of reconstruction error to determine the anomaly detection threshold.
## Visualization and Evaluation
After detecting anomalies, the project provides several forms of feedback and analysis:

-**Training Loss History:** Visualizes the model’s training and validation loss over epochs.
-**t-SNE Visualization:** Projects the high-dimensional transaction embeddings into 2D space using t-SNE, providing a visual representation of normal and fraudulent transactions.
-**Reconstruction Error Distribution:** Displays the distribution of reconstruction errors for both normal and fraudulent transactions.
-**Statistical Analysis:** Highlights the differences in feature values between predicted anomalies and normal transactions.
-**Performance Metrics:** The model's performance is evaluated using a confusion matrix and a classification report.

**Author**
This project was developed by           **RADHWANE BENAISSA**.
Contributions and suggestions are welcome.
Please feel free to raise issues or submit pull requests.