import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import L1L2

# Load the data
df = pd.read_csv("gmb.csv", sep=",", usecols=['Word', 'POS', 'Tag'])
df = df.dropna(subset=['Word', 'POS', 'Tag'])
df['Word'] = df['Word'].astype(str)


# Input and Output extraction
words = df['Word'].tolist()
pos_tags = df['POS'].tolist()
ne_tags = df['Tag'].tolist()

# Tokenize words
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(words)
X = word_tokenizer.texts_to_sequences(words)

# Tokenize POS and NE tags
pos_encoder = LabelEncoder()
pos_labels = pos_encoder.fit_transform(pos_tags)

ne_encoder = LabelEncoder()
ne_labels = ne_encoder.fit_transform(ne_tags)

# Pad sequences
MAX_LEN = 20  # Define a max length
X = pad_sequences(X, maxlen=MAX_LEN, padding='post')

# One-hot encode labels (POS and NE)
pos_labels = to_categorical(pos_labels)
ne_labels = to_categorical(ne_labels)

# Combine labels
y = np.concatenate([pos_labels, ne_labels], axis=1)  # Merge POS and NE labels for simplicity

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)

# Define regularizers
regularizer = L1L2(l1=1e-5, l2=1e-4)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(word_tokenizer.word_index)+1, output_dim=128, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3, 
                             kernel_regularizer=regularizer, recurrent_regularizer=regularizer)))
model.add(Dense(y.shape[1], activation='softmax'))  # Output layer for combined POS and NE tags

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=1, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
