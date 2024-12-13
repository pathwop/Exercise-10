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

# Load and preprocess data
df = pd.read_csv("gmb.csv", sep=",", encoding='latin1')
df["Word"] = df["Word"].astype(str)
df = df.dropna(subset=["Word", "Tag", "POS"]) 
df = df.head(2000)

X = df["Word"]
y_tag = df["Tag"]
y_pos = df["POS"]

y_tag, tag_classes = pd.factorize(y_tag)  
y_pos, pos_classes = pd.factorize(y_pos) 

X_train, X_test, y_tag_train, y_tag_test, y_pos_train, y_pos_test = train_test_split(
    X, y_tag, y_pos, test_size=0.1, random_state=42)

X_train = X_train.tolist()
X_test = X_test.tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_word) for tokenized_word in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding='post')
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding='post')

def bi_lstm_model(vocab_size, MAX_LENGTH, tag_classes, pos_classes):
    model_tag = Sequential()
    model_tag.add(Input(shape=(MAX_LENGTH,)))
    model_tag.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH))
    model_tag.add(Bidirectional(LSTM(64, activation="tanh", recurrent_activation="sigmoid", dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4))))
    model_tag.add(Flatten())
    model_tag.add(Dense(64, activation='tanh'))
    model_tag.add(Dropout(0.3))
    model_tag.add(Dense(len(tag_classes), activation='softmax', name="tag_output"))
    model_tag.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_pos = Sequential()
    model_pos.add(Input(shape=(MAX_LENGTH,)))
    model_pos.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH))
    model_pos.add(Bidirectional(LSTM(64, activation="tanh", recurrent_activation="sigmoid", dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4))))
    model_pos.add(Flatten())
    model_pos.add(Dense(64, activation='tanh'))
    model_pos.add(Dropout(0.3))
    model_pos.add(Dense(len(pos_classes), activation='softmax', name="tag_output"))
    model_pos.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model_tag, model_pos

def lstm_model(vocab_size, MAX_LENGTH, tag_classes, pos_classes):
    model_tag = Sequential()
    model_tag.add(Input(shape=(MAX_LENGTH,)))
    model_tag.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH))
    model_tag.add(LSTM(64, activation="tanh", recurrent_activation="sigmoid", dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4)))
    model_tag.add(Flatten())
    model_tag.add(Dense(64, activation='tanh'))
    model_tag.add(Dropout(0.3))
    model_tag.add(Dense(len(tag_classes), activation='softmax', name="tag_output"))
    model_tag.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_pos = Sequential()
    model_pos.add(Input(shape=(MAX_LENGTH,)))
    model_pos.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH))
    model_pos.add(LSTM(64, activation="tanh", recurrent_activation="sigmoid", dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4)))
    model_pos.add(Flatten())
    model_pos.add(Dense(64, activation='tanh'))
    model_pos.add(Dropout(0.3))
    model_pos.add(Dense(len(pos_classes), activation='softmax', name="tag_output"))
    model_pos.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model_tag, model_pos

def rnn_model(vocab_size, MAX_LENGTH, tag_classes, pos_classes):
    model_tag = Sequential()
    model_tag.add(Input(shape=(MAX_LENGTH,)))
    model_tag.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH))
    model_tag.add(SimpleRNN(64, activation="tanh", dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4)))
    model_tag.add(Flatten())
    model_tag.add(Dense(64, activation='tanh'))
    model_tag.add(Dropout(0.3))
    model_tag.add(Dense(len(tag_classes), activation='softmax', name="tag_output"))
    model_tag.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_pos = Sequential()
    model_pos.add(Input(shape=(MAX_LENGTH,)))
    model_pos.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH))
    model_pos.add(SimpleRNN(64, activation="tanh", dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4)))
    model_pos.add(Flatten())
    model_pos.add(Dense(64, activation='tanh'))
    model_pos.add(Dropout(0.3))
    model_pos.add(Dense(len(pos_classes), activation='softmax', name="tag_output"))
    model_pos.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model_tag, model_pos

# LSTM Output -> Tag: 0.44, POS: 0.53
model_tag, model_pos = lstm_model(vocab_size, MAX_LENGTH, tag_classes, pos_classes)

# Bidirectional LSTM Output -> Tag: 0.44, POS: 0.60
# model_tag, model_pos = bi_lstm_model(vocab_size, MAX_LENGTH, tag_classes, pos_classes)


# RNN Output -> Tag: 0.35, POS: 0.59
# model_tag, model_pos = rnn_model(vocab_size, MAX_LENGTH, tag_classes, pos_classes)


model_tag.fit(tokenized_X_train, y_tag_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
model_pos.fit(tokenized_X_train, y_pos_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

y_pred_tag = model_tag.predict(tokenized_X_test)
y_pred_pos = model_pos.predict(tokenized_X_test)

y_pred_tag = np.argmax(y_pred_tag, axis=1)
y_pred_pos = np.argmax(y_pred_pos, axis=1)

print("Tag Classification Report:")
print(classification_report(y_tag_test, y_pred_tag))

print("POS Classification Report:")
print(classification_report(y_pos_test, y_pred_pos))


