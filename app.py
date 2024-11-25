import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model and preprocessing tools
rf_model = joblib.load('rf_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
le_prediction = joblib.load('prediction_label_encoder.pkl')

# Streamlit app
st.title("Cyber Threat Prediction")
st.write("Upload your data or fill in the fields to predict cyber threat categories.")

# Feature input
st.sidebar.header("Input Features")
input_data = {
    "Time": st.sidebar.number_input("Time", value=10),
    "Protcol": st.sidebar.selectbox("Protocol", ["TCP", "UDP"]),
    "Flag": st.sidebar.selectbox("Flag", ["A", "B", "C"]),
    "Family": st.sidebar.selectbox("Family", ["WannaCry", "Petya", "NotPetya"]),
    "Clusters": st.sidebar.slider("Clusters", 1, 12, value=1),
    "SeddAddress": st.sidebar.text_input("Sender Address", value="1DA11mPS"),
    "ExpAddress": st.sidebar.text_input("Exp Address", value="1BonuSr7"),
    "BTC": st.sidebar.number_input("BTC", value=1),
    "USD": st.sidebar.number_input("USD", value=500),
    "Netflow_Bytes": st.sidebar.number_input("Netflow Bytes", value=500),
    "IPaddress": st.sidebar.text_input("IP Address", value="192.168.1.1"),
    "Threats": st.sidebar.selectbox("Threats", ["Bonet", "DDoS", "Phishing"]),
    "Port": st.sidebar.slider("Port", 5061, 5068, value=5061)
}

# Convert input to dataframe
input_df = pd.DataFrame([input_data])

# Preprocess input data
for col, le in label_encoders.items():
    if col in input_df:
        input_df[col] = le.transform(input_df[col])
        
# Standardize numerical columns
numerical_columns = input_df.select_dtypes(include="number").columns
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Predict button
if st.button("Predict"):
    prediction = rf_model.predict(input_df)
    st.success(f"The predicted threat is: {le_prediction.inverse_transform(prediction)[0]}")
