# -*- coding: utf-8 -*-
"""
Consolidated Python Script for AML Project
Automatically extracted and converted into a single script.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Use the TkAgg backend for displaying plots
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Install required libraries
#os.system("pip install openpyxl kmodes flask ngrok scikit-learn pandas")

# Load the dataset
file_path = "synthetic_banking_data.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Preprocessing
string_columns = ['feature_requests', 'complaint_topics']
for column in string_columns:
    if df[column].isnull().sum() > 0:
        df[column].fillna(df[column].mode()[0], inplace=True)

# K-Modes Clustering Example
X = df[['gender', 'occupation']]
kmodes = KModes(n_clusters=2, init='Cao', n_init=1, verbose=0)
df['cluster'] = kmodes.fit_predict(X)

# Visualize Clusters (Gender vs. Occupation)
sns.catplot(
    x='gender', y='occupation', hue='cluster', data=df, kind='strip',
    jitter=True, dodge=True, palette='Set1', height=5, aspect=1.2
)
plt.title('K-Modes Clustering (Gender vs. Occupation)')
plt.xlabel("Gender")
plt.ylabel("Occupation")
plt.show()

# PCA Example
numeric_columns = ['age', 'transaction_frequency', 'total_tx_volume',
                   'avg_tx_value', 'satisfaction_score', 'nps_score',
                   'active_products', 'churn_probability', 'customer_lifetime_value',
                   'total_transaction_volume', 'monthly_transaction_count', 'avg_daily_transactions']
categorical_columns = ['income_bracket', 'occupation', 'customer_segment', 'education_level',
                       'marital_status', 'acquisition_channel', 'savings_account',
                       'credit_card', 'personal_loan', 'investment_account']

scaler = StandardScaler()
scaled_numeric_data = scaler.fit_transform(df[numeric_columns])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical_data = encoder.fit_transform(df[categorical_columns])

combined_data = pd.concat([
    pd.DataFrame(scaled_numeric_data, columns=numeric_columns),
    pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))
], axis=1)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(combined_data)
df['pca_1'] = pca_features[:, 0]
df['pca_2'] = pca_features[:, 1]

# Visualize PCA Components
plt.scatter(df['pca_1'], df['pca_2'], alpha=0.7)
plt.title('PCA of All Attributes (2 Components)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Train-Test Split and Random Forest Regression
features = ['pca_1', 'pca_2', 'cluster']
target = 'churn_probability'
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Agglomerative Clustering
dendrogram(linkage(combined_data, method='ward'), orientation='top', labels=df.index.tolist())
plt.title('Dendrogram of Agglomerative Clustering')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()

# Cross-Sell Recommendation Example
def cross_sell_recommendation(customer):
    recommendations = []
    if customer['savings_account'] == 0 and customer['transaction_frequency'] > 10:
        recommendations.append('Open a Savings Account')
    if customer['personal_loan'] == 0 and customer['avg_daily_transactions'] > 5:
        recommendations.append('Apply for a Personal Loan')
    if customer['customer_lifetime_value'] > 5000 and customer['investment_account'] == 0:
        recommendations.append('Start an Investment Account')
    if customer['churn_probability'] > 0.7:
        recommendations.append('Explore other products to enhance engagement')
    return recommendations

df['recommendations'] = df.apply(cross_sell_recommendation, axis=1)
print(df[['customer_id', 'recommendations']].head())

# End of Script
