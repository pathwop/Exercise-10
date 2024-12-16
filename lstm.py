import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load dataset
df = pd.read_csv("gmb.csv", encoding="ISO-8859-1")
df = df.fillna(method="ffill")  # Fill NaN values

# Preprocessing
def preprocess_data(df):
    words = list(set(df["Word"].values))
    tags = list(set(df["Tag"].values))

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    sentences = []
    sentence_tags = []

    for sentence in df.groupby("Sentence #"):
        sentences.append([word2idx[w] for w in sentence[1]["Word"].values])
        sentence_tags.append([tag2idx[t] for t in sentence[1]["Tag"].values])

    return sentences, sentence_tags, word2idx, tag2idx, idx2tag

sentences, sentence_tags, word2idx, tag2idx, idx2tag = preprocess_data(df)

# Padding
MAXLEN = 50  # Maximum sentence length
num_words = len(word2idx) + 1
num_tags = len(tag2idx)

X = pad_sequences(sentences, maxlen=MAXLEN, padding="post")
y = pad_sequences(sentence_tags, maxlen=MAXLEN, padding="post")
y = [to_categorical(i, num_classes=num_tags) for i in y]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define Models
def build_model(model_type="LSTM"):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=128, input_length=MAXLEN))

    if model_type == "LSTM":
        model.add(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    elif model_type == "BiLSTM":
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    elif model_type == "RNN":
        model.add(SimpleRNN(64, return_sequences=True, dropout=0.3))

    model.add(Dense(num_tags, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Train and Evaluate Models
for model_type in ["LSTM", "BiLSTM", "RNN"]:
    print(f"\nTraining {model_type} model...")
    model = build_model(model_type)
    model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

    # Evaluation
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=-1).flatten()
    y_true_classes = np.argmax(np.array(y_test), axis=-1).flatten()

    print(f"Results for {model_type}:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=list(tag2idx.keys())))
