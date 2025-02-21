import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

# Initialize models and preprocessors
rf_model = RandomForestRegressor(random_state=42)
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

def initialize_models():
    global rf_model, scaler, encoder

    # Create a larger synthetic dataset for proper initialization
    n_samples = 1000

    # Simulated numeric data
    numeric_features = np.random.rand(n_samples, 5)

    # Create categorical features as strings
    categorical_features = np.array([['No', 'No', 'No'] for _ in range(n_samples)])

    # Fit the preprocessors
    scaler.fit(numeric_features)
    encoder.fit(categorical_features)

    # Generate synthetic target values
    y = np.random.rand(n_samples)

    # Prepare features for model training
    encoded_categorical = encoder.transform(categorical_features)
    scaled_numeric = scaler.transform(numeric_features)
    combined_features = np.hstack([scaled_numeric, encoded_categorical])

    # Train the model
    rf_model.fit(combined_features, y)

initialize_models()

def display_recommendations(recommendations):
    if not recommendations:
        st.write("No recommendations at this time.")
    else:
        st.write("Recommended products for this customer:")
        for i, recommendation in enumerate(recommendations, 1):  # Start counting from 1
            st.write(f"{i}. {recommendation}")

def cross_sell_recommendation(data):
    recommendations = []
    if data.get('savings_account') == 'No' and data.get('transaction_frequency', 0) > 10:
        recommendations.append('Open a Savings Account')
    if data.get('personal_loan') == 'No' and data.get('avg_daily_transactions', 0) > 5:
        recommendations.append('Apply for a Personal Loan')
    if data.get('customer_lifetime_value', 0) > 5000 and data.get('investment_account') == 'No':
        recommendations.append('Start an Investment Account')
    return recommendations

st.title("AML Interface: Recommendations and Predictions")

st.sidebar.header("Input Features")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
transaction_frequency = st.sidebar.number_input("Transaction Frequency", min_value=0, value=15)
total_tx_volume = st.sidebar.number_input("Total Transaction Volume", min_value=0, value=5000)
avg_daily_transactions = st.sidebar.number_input("Average Daily Transactions", min_value=0, value=2)
customer_lifetime_value = st.sidebar.number_input("Customer Lifetime Value", min_value=0, value=10000)
savings_account = st.sidebar.selectbox("Savings Account", ['No', 'Yes'])
personal_loan = st.sidebar.selectbox("Personal Loan", ['No', 'Yes'])
investment_account = st.sidebar.selectbox("Investment Account", ['No', 'Yes'])

input_data = {
    "age": age,
    "transaction_frequency": transaction_frequency,
    "total_tx_volume": total_tx_volume,
    "avg_daily_transactions": avg_daily_transactions,
    "customer_lifetime_value": customer_lifetime_value,
    "savings_account": savings_account,
    "personal_loan": personal_loan,
    "investment_account": investment_account,
}

st.subheader("Input Data")
st.write(pd.DataFrame([input_data]))

st.subheader("Cross-Sell Recommendations")
recommendations = cross_sell_recommendation(input_data)
display_recommendations(recommendations)

try:
    # Prepare numeric features
    numeric_input = np.array([[age, transaction_frequency, total_tx_volume,
                               avg_daily_transactions, customer_lifetime_value]])
    scaled_numeric = scaler.transform(numeric_input)

    # Prepare categorical features
    categorical_input = np.array([[savings_account, personal_loan, investment_account]])
    encoded_categorical = encoder.transform(categorical_input)

    # Combine features
    combined_features = np.hstack([scaled_numeric, encoded_categorical])

    # Make prediction
    prediction = rf_model.predict(combined_features)[0]

    st.subheader("Predicted Churn Probability")
    st.write(f"{prediction:.2f}")

    # Create visualization of feature importance
    feature_importance = pd.DataFrame({
        'Feature': ['Age', 'Transaction Freq', 'TX Volume', 'Daily TX', 'CLV'] +
                   [f'Cat_{i}' for i in range(encoded_categorical.shape[1])],
        'Importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error during prediction: {str(e)}")
    st.error("Please check the input data format and try again.")
