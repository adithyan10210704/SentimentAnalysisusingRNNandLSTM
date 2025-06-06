import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load the IMDb dataset
df = pd.read_csv("data/IMDB Dataset.csv")  # Ensure this file exists in the data/ folder

# Clean the text (optional but helpful)
df['review'] = df['review'].apply(lambda x: re.sub('<.*?>', '', x))  # Remove HTML tags
df['review'] = df['review'].apply(lambda x: re.sub('[^a-zA-Z ]', '', x.lower()))  # Remove non-alpha chars and lowercase

# Convert sentiment to binary
y = np.array([1 if sentiment.strip() == "positive" else 0 for sentiment in df['sentiment']])

# Tokenize the reviews
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(sequences, maxlen=100)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RNN model
model_rnn = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model_rnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model_rnn.summary())

# Train model
model_rnn.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate on test set
loss_rnn, acc_rnn = model_rnn.evaluate(X_test, y_test)
print(f"RNN Test Accuracy: {acc_rnn:.2f}")

# Test prediction
text = ["This movie was so bad"]
seq = tokenizer.texts_to_sequences(text)
padded = pad_sequences(seq, maxlen=100)
prediction = model_rnn.predict(padded)
print("Sentiment Score:", prediction[0][0])
print("Predicted Sentiment:", "Positive" if prediction[0][0] >= 0.5 else "Negative")

# Save model
model_rnn.save("models/sentiment_rnn_model.h5")

# Reload model (optional)
loaded_model = load_model("models/sentiment_rnn_model.h5")
