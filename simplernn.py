import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.preprocessing import sequence


## Load imdb dataset
max_features = 10000 ## vocabulary size
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape:{X_test.shape}, Testing labels shape: {y_test.shape}")

## inspect a sample and its labels
sample_review = X_train[0]
sample_label = y_train[0]
print(f"Sample review as integers{sample_review}")
print(f"Sample label as integers{sample_label}")

## Mapping of words index back to words

word_index = imdb.get_word_index()
print(word_index)
reverse_word_index = {value:key for key, value in word_index.items()}
print(reverse_word_index)
decoded_review = ' '.join(reverse_word_index.get(i - 3, '?') for i in sample_review)
print(decoded_review)

## apply prepadding

X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)
print(X_train)
print(X_test)

## create a simple RNN
model = Sequential()
dim=128
max_length=500
model.add(Embedding(max_features,dim,input_length = max_length)) ## embedding layers
model.add(SimpleRNN(128,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.build((None, max_length))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

## Create an instance of EarlyStopping callback

from tensorflow.keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

## Train the mode with early stopping

history = model.fit(
    X_train,y_train, epochs = 10, batch_size = 32,validation_split = 0.2,
    callbacks = [earlystopping]
)

## save the model file
model.save('simple_rnn_imdb.h5')