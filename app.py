import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("Fake News Detection Demo")
st.write("This application uses a BERT model to detect fake news articles.")

user_input = st.text_area("Enter news article text:", height=200)

if user_input:
    with st.spinner('Analyzing text...'):
        model = pipeline("text-classification", model="bert-base-uncased")
        result = model(user_input)
        
        # Display result
        prediction = "REAL" if result[0]['label'] == "LABEL_1" else "FAKE"
        confidence = round(result[0]['score'] * 100, 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Confidence", f"{confidence}%")
            
        if confidence < 70:
            st.warning("⚠️ The model's confidence is relatively low. Please verify from other sources.")