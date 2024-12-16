import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, SimpleRNN, LSTM, Bidirectional, TimeDistributed, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("gmb.csv", sep=",")

def preprocess_data(df):
    words = list(set(df['Word'].values))
    pos_tags = list(set(df['POS'].values))
    ne_tags = list(set(df['NE'].values))

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    pos2idx = {p: i for i, p in enumerate(pos_tags)}
    ne2idx = {n: i for i, n in enumerate(ne_tags)}

    idx2pos = {i: p for p, i in pos2idx.items()}
    idx2ne = {i: n for n, i in ne2idx.items()}

    df["Word_idx"] = df["Word"].map(word2idx)
    df["POS_idx"] = df["POS"].map(pos2idx)
    df["NE_idx"] = df["NE"].map(ne2idx)

    grouped = df.groupby("Sentence #").agg(
        {
            "Word_idx": list,
            "POS_idx": list,
            "NE_idx": list,
        }
    )

    return grouped, word2idx, pos2idx, ne2idx, idx2pos, idx2ne

grouped, word2idx, pos2idx, ne2idx, idx2pos, idx2ne = preprocess_data(df)

MAXLEN = 50
X = pad_sequences(grouped["Word_idx"], maxlen=MAXLEN, padding="post")
Y_POS = pad_sequences(grouped["POS_idx"], maxlen=MAXLEN, padding="post")
Y_NE = pad_sequences(grouped["NE_idx"], maxlen=MAXLEN, padding="post")

Y_POS = np.array([to_categorical(i, num_classes=len(pos2idx)) for i in Y_POS])
Y_NE = np.array([to_categorical(i, num_classes=len(ne2idx)) for i in Y_NE])

X_train, X_test, Y_POS_train, Y_POS_test, Y_NE_train, Y_NE_test = train_test_split(
    X, Y_POS, Y_NE, test_size=0.1, random_state=42
)

def build_model(word2idx, output_dim_pos, output_dim_ne, model_type="LSTM"):
    model = Sequential()
    model.add(Embedding(input_dim=len(word2idx), output_dim=128, input_length=MAXLEN))

    if model_type == "LSTM":
        model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    elif model_type == "BiLSTM":
        model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    elif model_type == "RNN":
        model.add(SimpleRNN(256, return_sequences=True, dropout=0.3))

    model.add(TimeDistributed(Dense(output_dim_pos, activation="softmax"), name="POS"))

    model.add(TimeDistributed(Dense(output_dim_ne, activation="softmax"), name="NE"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

def train_and_evaluate(model_type):
    model = build_model(word2idx, len(pos2idx), len(ne2idx), model_type)

    history = model.fit(
        X_train, {"POS": Y_POS_train, "NE": Y_NE_train},
        validation_data=(X_test, {"POS": Y_POS_test, "NE": Y_NE_test}),
        batch_size=64,
        epochs=5,
        verbose=1
    )

    predictions = model.predict(X_test)
    pos_pred = np.argmax(predictions["POS"], axis=-1).flatten()
    ne_pred = np.argmax(predictions["NE"], axis=-1).flatten()

    pos_true = np.argmax(Y_POS_test, axis=-1).flatten()
    ne_true = np.argmax(Y_NE_test, axis=-1).flatten()

    print("POS Classification Report:")
    print(classification_report(pos_true, pos_pred, target_names=list(pos2idx.keys())))

    print("NE Classification Report:")
    print(classification_report(ne_true, ne_pred, target_names=list(ne2idx.keys())))

print("LSTM Results:")
train_and_evaluate("LSTM")

print("BiLSTM Results:")
train_and_evaluate("BiLSTM")

print("RNN Results:")
train_and_evaluate("RNN")
