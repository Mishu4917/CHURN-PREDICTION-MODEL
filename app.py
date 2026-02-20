import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# LOAD SAVED FILES
# ---------------------------
model = joblib.load("churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details below:")

# ---------------------------
# USER INPUT SECTION
# ---------------------------

user_input = {}

for feature in feature_order:
    user_input[feature] = st.text_input(f"Enter {feature}")

# ---------------------------
# PREDICT BUTTON
# ---------------------------

if st.button("Predict Churn"):

    input_df = pd.DataFrame([user_input])

    # Apply Label Encoding
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(
                input_df[col].astype(str)
            )

    input_df = input_df[feature_order]

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ This customer is likely to CHURN")
    else:
        st.success("✅ This customer is NOT likely to churn")
