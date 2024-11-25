import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Title and description
st.title("Cyber Threat Detection")
st.write("Predict whether a threat is SS, A, or S based on provided parameters.")

# Input fields for features
features = {
    "Time": st.number_input("Time", min_value=0),
    "Protcol": st.selectbox("Protocol", list(label_encoders['Protcol'].classes_)),
    "Flag": st.selectbox("Flag", list(label_encoders['Flag'].classes_)),
    "Family": st.selectbox("Family", list(label_encoders['Family'].classes_)),
    "Clusters": st.number_input("Clusters", min_value=0),
    "SeddAddress": st.selectbox("SeddAddress", list(label_encoders['SeddAddress'].classes_)),
    "ExpAddress": st.selectbox("ExpAddress", list(label_encoders['ExpAddress'].classes_)),
    "BTC": st.number_input("BTC", min_value=0),
    "USD": st.number_input("USD", min_value=0),
    "Netflow_Bytes": st.number_input("Netflow_Bytes", min_value=0),
    "IPaddress": st.selectbox("IPaddress", list(label_encoders['IPaddress'].classes_)),
    "Threats": st.selectbox("Threats", list(label_encoders['Threats'].classes_)),
    "Port": st.number_input("Port", min_value=0),
}

# Convert categorical inputs
for key in features:
    if key in label_encoders:
        features[key] = label_encoders[key].transform([features[key]])[0]

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)
    prediction_label = target_encoder.inverse_transform(prediction)[0]
    st.write(f"The predicted threat is: *{prediction_label}*")
