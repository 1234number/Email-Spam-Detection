import streamlit as st 
import pandas as pd
import joblib

model =joblib.load("Spam_Email_Prediction.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Email Spam Detection")
st.write("Enter the Email Details below to predict Whether it is Spam or Not")

email_length = st.number_input(
    "Email Length",
    min_value=0,
    value=100
)

num_links = st.number_input(
    "Number of Links",
    min_value=0,
    value=1
)

num_special_chars = st.number_input(
    "Number of Special Characters",
    min_value=0,
    value=2
)

capital_words = st.number_input(
    "Capital Words",
    min_value=0,
    value=5
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Email_Length": [email_length],
        "Num_Links": [num_links],
        "Num_Special_Chars": [num_special_chars],
        "Capital_Words": [capital_words]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("🚨 Spam Email")
    else:
        st.success("✅ Not Spam")
