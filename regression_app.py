import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
import datetime
import streamlit as st
import numpy as np
## Load the model

model = tf.keras.models.load_model('model.h5')

## Load encoders and scaler

with open ('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## Streamlit app
st.title("Customer Salary Prediction")

## inputs for user to give

geography = st.selectbox('Select Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Select Gender',label_encoder.classes_)
CreditScore = st.number_input('Credit Score')
age = st.slider('Select Age',18,92)
balance = st.number_input('Balance')
Exited = int(st.checkbox('Exited?'))
tenure = st.slider('Tenure',0,10)
NumOfProducts = st.slider('Num of Products',1,4)
HasCrCard = int(st.checkbox('Has CR Card'))
IsActiveMember = int(st.checkbox('Is Active'))

## take input in the form of dictionary

input_data = {
    'CreditScore': [CreditScore],
    'Geography':[geography],
    'Gender':[gender],
    'Age': [age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'Exited':[Exited]
}
## convert input dictionary into dataframe
input_data = pd.DataFrame(input_data)
## label encode gender
input_data['Gender'] = label_encoder.transform(input_data['Gender'])

## one hot encode geography

geo_encoded = onehot_encoder.transform(input_data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns = onehot_encoder.get_feature_names_out(['Geography']))

## drop old geography column and concat

input_df = pd.concat([input_data.drop(['Geography'],axis=1), geo_encoded_df], axis=1)

## Scale the input

input_scaled = scaler.transform(input_df)

## predict
prediction = model.predict(input_scaled)
print("Prediction", prediction)
print(f"Prediction Salary: {prediction[0][0]}")
## Write the prediction on Streamlist app
st.write("Prediction Salary::", prediction[0][0])






