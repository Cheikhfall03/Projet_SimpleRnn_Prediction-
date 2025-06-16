import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('model_review_rnn.h5',compile=False)

# Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

# Custom CSS to improve the design
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .stTitle {
            color: #0066cc;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
        }
        .stTextArea {
            background-color: #fff;
            border: 2px solid #0066cc;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px;
        }
        .stButton {
            background-color: #0066cc;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton:hover {
            background-color: #004d99;
        }
        .stWrite {
            font-size: 20px;
            color: #333333;
            text-align: center;
            margin-top: 20px;
        }
        .result {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .prediction-score {
            font-size: 16px;
            text-align: center;
            margin-top: 10px;
            color: #555555;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="stTitle">IMDB Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)
st.write('Enter a movie review below to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review', height=200, max_chars=1000)

# Button for classification
if st.button('Classify', key="classify_button"):
    if user_input:
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display results with styling
        st.markdown(f'<p class="result">Sentiment: {sentiment}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-score">Prediction Score: {prediction[0][0]:.4f}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="stWrite">Please enter a movie review to classify.</p>', unsafe_allow_html=True)

