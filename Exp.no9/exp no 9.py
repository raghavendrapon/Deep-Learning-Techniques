# Step 1: Import Required Libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Step 2: Load Dataset
vocab_size = 10000  # Limit to top 10,000 words
max_len = 200       # Max review length

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Step 3: Preprocess - Pad sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Step 4: Build RNN Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len))  # Word vector embedding
model.add(SimpleRNN(units=64))  # Recurrent layer to learn sequence patterns
model.add(Dense(1, activation='sigmoid'))  # Binary output: 0 (neg) or 1 (pos)

# Step 5: Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train Model
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)

# Step 7: Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")
