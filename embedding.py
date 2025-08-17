from tensorflow.keras.preprocessing.text import one_hot

text = ['the glass of milk', 'the glass of juice', 'I am a good boy',
        'I am a good developer', 'understand the meaning of words',
        'your videos are good'
]

## Define vocabulary size
voc_size = 10000

## One hot representation

one_hot_representation = [one_hot(words,voc_size) for words in text]
print(one_hot_representation)

## consider the first list in the output
## it is [7052, 5900, 4788, 7098]
## this is vector index representation of the glass of milk
## the is in 7052 position, glass in 5900, of in 4788, milk in 7098
## what that means is there is one vector for the with 10000 rows and at index 7052 the is there or it is 1 whereas rest of the rows are zero


## word embedding representation

from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
import numpy as np


## pad_sequences are needed to make all the sentences to equal length as for RNN it should be of equal length
## you can decide a max length based on the length of the sentences, it just adds zero to make sentences to same length
max_length = 8
embedded_docs = pad_sequences(one_hot_representation,padding="pre",maxlen=max_length)
## pre gives zeros in beginning post gives zeros at the end
print(embedded_docs)

## feature representation
dim = 10
model = Sequential()
model.add(Embedding(voc_size, dim, input_length=max_length))
model.build((None, max_length))
model.compile(loss='mse', optimizer='adam')
print(model.summary())
print(model.predict(embedded_docs)) ## will show each word with 10 dimensions for all sentences
print(model.predict(embedded_docs[:1])) ## will show each word with 10 dimensions for the first senstence


