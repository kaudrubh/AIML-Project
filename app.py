import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained model and preprocessing tools
rf_model = joblib.load('rf_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
le_prediction = joblib.load('prediction_label_encoder.pkl')

# Streamlit app
st.title("Cyber Threat Prediction")
st.write("Upload your data or fill in the fields to predict cyber threat categories.")

# Sidebar for input features
st.sidebar.header("Input Features")

# Use the label encoders' classes to populate dropdown options
protocol_options = label_encoders["Protcol"].classes_
flag_options = label_encoders["Flag"].classes_
family_options = label_encoders["Family"].classes_
threat_options = label_encoders["Threats"].classes_

# Input fields
input_data = {
    "Time": st.sidebar.number_input("Time", value=10),
    "Protcol": st.sidebar.selectbox("Protocol", protocol_options),
    "Flag": st.sidebar.selectbox("Flag", flag_options),
    "Family": st.sidebar.selectbox("Family", family_options),
    "Clusters": st.sidebar.slider("Clusters", 1, 12, value=1),
    "SeddAddress": st.sidebar.text_input("Sender Address", value="1DA11mPS"),
    "ExpAddress": st.sidebar.text_input("Exp Address", value="1BonuSr7"),
    "BTC": st.sidebar.number_input("BTC", value=1),
    "USD": st.sidebar.number_input("USD", value=500),
    "Netflow_Bytes": st.sidebar.number_input("Netflow Bytes", value=500),
    "IPaddress": st.sidebar.text_input("IP Address", value="192.168.1.1"),  # Pass unchanged
    "Threats": st.sidebar.selectbox("Threats", threat_options),
    "Port": st.sidebar.slider("Port", 5061, 5068, value=5061)
}

# Convert input to dataframe
input_df = pd.DataFrame([input_data])

# Preprocess input data
try:
    # Encode categorical features, excluding columns like 'IPaddress'
    for col, le in label_encoders.items():
        if col in input_df and col != "IPaddress":  # Exclude IPaddress from encoding
            if input_df[col][0] not in le.classes_:
                st.error(f"Invalid value for {col}: {input_df[col][0]}. Please provide a valid value.")
                st.stop()  # Stop execution if there's an invalid input
            input_df[col] = le.transform(input_df[col])

    # Standardize numerical columns
    numerical_columns = input_df.select_dtypes(include="number").columns
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Ensure all required features are present
    expected_features = rf_model.feature_names_in_  # Ensure the features match the training phase
    missing_features = set(expected_features) - set(input_df.columns)
    if missing_features:
        for feature in missing_features:
            input_df[feature] = 0  # Add missing features with default values
    input_df = input_df[expected_features]  # Reorder columns to match the model input

    # Predict button
    if st.button("Predict"):
        prediction = rf_model.predict(input_df)
        predicted_class = le_prediction.inverse_transform(prediction)[0]
        st.success(f"The predicted threat is: {predicted_class}")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
