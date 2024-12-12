import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("gmb.csv", sep=",", usecols=['Word', 'POS', 'Tag'])
df = df.head(1000)
X = df['Word']
y= df[['POS', 'Tag']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)
X_train = X_train.to_list()
X_test = X_test.to_list()
X_train = pad_sequences(X_train, padding='post')
X_test = pad_sequences(X_test, padding='post')
print(df['POS'].value_counts())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

regularizer = L1L2(l1=1e-5, l2=1e-4)
model = Sequential()
model.add(LSTM(64, activation="tanh", reccurrent_activation="sigmoid", dropout=0.3, reccurrent_dropout=0.3, return_sequences=True,
               kernal_regularizer=L1L2, bias_regularizer=L1L2, activity_regularizer=L1L2, reccurrent_regularizer=L1L2))
model.add(Bidirectional(LSTM(64)))
model.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=1)