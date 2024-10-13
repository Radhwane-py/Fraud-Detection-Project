import pandas as pd

df = pd.read_csv(r"D:\Fraud Detection Project Main\THE  AUTOENCODER  MODEL\Results\statistical_analysis.csv")

print(df.head(20))
print(df.info())
print(df.describe())
