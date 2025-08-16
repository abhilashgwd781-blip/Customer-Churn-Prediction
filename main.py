import pandas as pd
import tensorboard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import tensorflow as tf
import datetime
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model


# Load the data

data=pd.read_csv("Churn_Modelling.csv")
print(data.head())

# Pre-Process the data
## drop irrelevant features/columns
data= data.drop(["RowNumber","Surname","CustomerId"], axis=1, inplace=True) # axis = 0 will drop rows and axis = 1 drops columns, inplace = true will give new dataframe
print(data.head())

# Encoding categorical variables
# basically a variable that can be turned into 0 and 1 like Gender, male and female

label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
print(data.head())

# what if there are more than 2 categories in the categorical variable? then?
# Inc case of geography we would have Germany, France, India etc. and if we use Label encoder
# then Germany would be 0, France would be 1, India would be 2, the ANN would interpret that India>France>Germany
# So we will use One hot encoding for geography column as it would be 1s and 0s

#One hot encoding for geography column
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])
print(f"The one hot Encoded:{geo_encoder}")

# let's print the categories of the feature geography
print(onehot_encoder_geo.get_feature_names_out(['Geography']))

# let's print the encoded geography array as a dataframe

geo_encoder_df = pd.DataFrame(geo_encoder.toarray(),columns = onehot_encoder_geo.get_feature_names_out(['Geography']))
print(geo_encoder_df.head())


# now let us combine the encoded data back into original data
data = pd.concat([data.drop(['Geography'], axis=1), geo_encoder_df],axis=1)
print(data.head())

# save the encoders into a pickle file for later use

with open ('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open ('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)

# let us divide the data into independent and dependent features
# Exited is a dependent column and rest are independent columns
X= data.drop(['Exited'], axis=1) # independent
y = data['Exited'] # dependent
print(X.head())
print(y.head())

# split the data into a training and testing set cause we need to train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)
print(X_train.shape)
print(X_test.shape)
print(X_train.shape[1])

# save the scaled features for later use in pickle format

with open ('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# build our ANN Model
model = Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), # hidden layer one. with 64 neurons, and each neuron should be given an activation function of choice, and here it is relu, and the first hidden layer must be connected to input. it should be provided to input_shape, in our case X_Train.Shape[1] will give 12 inputs and , empty is used to make it a single dimension tuple with 12 inputs
    Dense(32,activation='relu'), # Hidden layer two with 32 neurons, and activation relu, no need of any shape as it is a sequential model, it is taken care of
    Dense(1,activation='sigmoid') # output layer with 1 neuron and sigmoid activation function
])

# let's get a summary of all the parameters, weights etc,

print(model.summary())

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy']) # for non binary classification you can use what is applicable

# Setup the TensorBoard to visualise the logs from training

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

# Set up early stopping
# It is needed to stop training early in case the loss is not reducing, so there is no need to train for 100s of epochs
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
# we are stopping based on loss value, patience indicates the number of epoches for stopping, and restore the best weights from all the epochs

## Training the model
history = model.fit(
    X_train,y_train, validation_data=(X_test,y_test), epochs=100,
    callbacks=[tensorboard_callback, early_stopping_callback]
)
print(history)
model.save('model.h5') #h5 is compatible with Keras
# if the training stops at a low epoch, then maybe increase patience
# for non legacy format save it as .keras
model.save('model.keras')

# Load Tensorboard extension
# tensorboard --logdir=logs/fit run this command in terminal

## Load the pickle files and do prediction using the model

model = load_model('model.h5')
## load the encoder adn scaler

with open("onehot_encoder_geo.pkl",'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl",'rb') as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)

## Example input data
input_data = {
    'CreditScore':600,
    'Geography':'France',
    'Gender':'Male',
    'Age': 40,
    'Tenure':3,
    'Balance':60000,
    'NumOfProducts':2,
    'HasCrCard':1,
    'IsActiveMember':1,
    'EstimatedSalary':50000
}
## Convert dictionary to dataframe
input_df = pd.DataFrame([input_data])
## apply label encoding for gender
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
## apply one hot encoding to geography
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']])
geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

## Drop old Geography and join encoded one
input_df = pd.concat([input_df.drop(['Geography'], axis=1), geo_encoded_df], axis=1)

## Scale numerical features
input_scaled = scaler.transform(input_df)

## predict
prediction = model.predict(input_scaled)
print("Prediction", prediction)
print(f"Prediction probability of churn: {prediction[0][0]}")
if prediction[0][0] > 0.5:
    print("Churn was successful")
else:
    print("Churn was not successful")

