import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_pipeline.pkl")

st.title("Telco Customer Churn Prediction")

# ====== USER INPUTS ======

gender = st.selectbox("Gender", ["Male", "Female"])

senior_citizen = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (Months)", min_value=0)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox("Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

monthly_charges = st.number_input("Monthly Charges")

# ====== PREDICT BUTTON ======

if st.button("Predict"):

    total_charges = tenure * monthly_charges

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prob = model.predict_proba(input_data)[0][1]

    st.success(f"Churn Probability: {round(prob*100,2)}%")


    if prob > 0.3:
        st.error("High Risk Customer")
    else:
        st.success("Low Risk Customer")
