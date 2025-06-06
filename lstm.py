import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the IMDb dataset
df = pd.read_csv("data/IMDB Dataset.csv")  # Ensure this file exists in the data/ folder

# Clean the reviews
df['review'] = df['review'].apply(lambda x: re.sub('<.*?>', '', x))  # Remove HTML
df['review'] = df['review'].apply(lambda x: re.sub('[^a-zA-Z ]', '', x.lower()))  # Remove punctuation and lowercase

# Convert sentiment to binary (positive = 1, negative = 0)
y = np.array([1 if sentiment.strip() == "positive" else 0 for sentiment in df['sentiment']])

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(sequences, maxlen=100)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model_lstm = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model_lstm.summary())

# Train the model
model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss_lstm, acc_lstm = model_lstm.evaluate(X_test, y_test)
print(f"LSTM Test Accuracy: {acc_lstm:.2f}")

# Predict on custom input
text = ["This movie was so bad"]
seq = tokenizer.texts_to_sequences(text)
padded = pad_sequences(seq, maxlen=100)
prediction = model_lstm.predict(padded)
print("Sentiment Score:", prediction[0][0])
print("Predicted Sentiment:", "Positive" if prediction[0][0] >= 0.5 else "Negative")

# Save the model
model_lstm.save("models/sentiment_lstm_model.h5")

# Load model (optional)
loaded_model = load_model("models/sentiment_lstm_model.h5")