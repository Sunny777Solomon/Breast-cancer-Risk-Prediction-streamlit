import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("lr_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # saved during training

st.title("🔬 Breast Cancer 10-Year Mortality Prediction (Logistic Regression)")

st.write("This app predicts the 10-year mortality risk for breast cancer patients using clinical and treatment features.")

# Collect user input
input_data = {}

for feature in feature_columns:
    # You can customize input types depending on the feature
    if "Age" in feature:
        input_data[feature] = st.number_input(f"{feature}", min_value=20, max_value=100, value=50)
    elif "Size" in feature or "Count" in feature:
        input_data[feature] = st.number_input(f"{feature}", min_value=0, max_value=200, value=10)
    elif "Status" in feature or "Therapy" in feature or "Stage" in feature:
        input_data[feature] = st.selectbox(f"{feature}", [0, 1])
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame in correct order
input_df = pd.DataFrame([input_data])[feature_columns]

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # probability of mortality (class 1)

    if prediction == 1:
        st.error(f"⚠️ Predicted: High risk of 10-year mortality.\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Predicted: Likely survival beyond 10 years.\n\nProbability: {prob:.2f}")
