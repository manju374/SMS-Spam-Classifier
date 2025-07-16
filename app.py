import streamlit as st
import joblib

model = joblib.load('spam_classifier_model.pkl')

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📧")

st.title("📧 SMS / Email Spam Classifier")
st.write("Enter the message below to detect if it's **Spam** or **Not Spam (Ham)**.")

input_message = st.text_area("Message")

if st.button("Predict"):
    result = model.predict([input_message])[0]
    if result == 1:
        st.error("🚫 This message is **Spam**.")
    else:
        st.success("✅ This message is **Not Spam (Ham)**.")
