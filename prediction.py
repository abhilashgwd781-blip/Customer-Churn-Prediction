import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## Mapping of words index back to words

word_index = imdb.get_word_index()
print(word_index)
reverse_word_index = {value:key for key, value in word_index.items()}
print(reverse_word_index)

# load the pre-trained model with ReLU Activation
model = load_model('simple_rnn_imdb.h5')
print(model.summary())

# helper functions

# function to decode reviews
def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+ 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
## prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# step 4 test with example
example_review = "Good movie, fantastic direction, splendid actors, nice costumes"

sentiment, prediction = predict_sentiment(example_review)
print(f"Review: {example_review}")
print(f"Sentiment: {sentiment}")
print(f"Prediction: {prediction}")

st.title("Simple RNN imdb prediction")
example_review = st.text_input("Review")
sentiment, prediction = predict_sentiment(example_review)
st.write("Review:", example_review)
st.write("Sentiment:", sentiment)
st.write("Prediction:", prediction)





