import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Create synthetic time series data
time_steps = 100
data = np.sin(np.linspace(0, 100, time_steps))
X = []
y = []
window_size = 10

for i in range(len(data) - window_size):
    X.append(data[i:i+window_size])
    y.append(data[i+window_size])

X = np.array(X)
y = np.array(y)

# Reshape data for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define RNN model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=8, verbose=0)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions[:5])
