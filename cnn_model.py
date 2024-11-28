from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
# Function to create a simple CNN model
def create_simple_cnn(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
    return model