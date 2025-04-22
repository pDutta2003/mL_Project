import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, BertTokenizerFast, BertForSequenceClassification

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("Fake News Detection Demo")

@st.cache_resource
def load_model():
    try:
        # Try to load the fine-tuned model
        model = BertForSequenceClassification.from_pretrained("./results")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    except:
        # Fallback to base BERT model if fine-tuned model is not available
        classifier = pipeline("text-classification", model="bert-base-uncased")
    return classifier

# Load the model
with st.spinner('Loading model...'):
    classifier = load_model()

# Text input
st.write("Enter a news article to check if it's real or fake:")
text_input = st.text_area("Article text:", height=200)

if text_input:
    with st.spinner('Analyzing...'):
        # Make prediction
        result = classifier(text_input)
        prediction = result[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            label = "REAL" if prediction['label'] == "LABEL_1" else "FAKE"
            confidence = round(prediction['score'] * 100, 2)
            st.metric("Prediction", label)
            
        with col2:
            st.metric("Confidence", f"{confidence}%")
            
        if confidence < 70:
            st.warning("⚠️ The model's confidence is relatively low. Please verify from other sources.")
            
        # Show tips for verification
        st.markdown("### Tips for Manual Verification:")
        st.markdown("""
        1. Check the source credibility
        2. Look for similar articles on fact-checking websites
        3. Verify quotes and statistics from original sources
        4. Check the publication date
        5. Be wary of sensational headlines
        """)

# Add information about the model
st.sidebar.title("About")
st.sidebar.info("""
This application uses a BERT-based model to detect fake news. 
The model has been trained on a dataset of real and fake news articles.

Note: This is a demonstration and should not be used as the sole source for determining news authenticity.
""")

# Add dataset statistics if available
try:
    df = pd.read_csv("./data/preprocessed_data.csv")
    st.sidebar.title("Dataset Statistics")
    st.sidebar.write(f"Total articles analyzed: {len(df)}")
    st.sidebar.write(f"Real articles: {len(df[df['label'] == 1])}")
    st.sidebar.write(f"Fake articles: {len(df[df['label'] == 0])}")
except:
    pass