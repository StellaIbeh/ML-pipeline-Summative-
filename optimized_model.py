from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l1
from keras.optimizers import RMSprop
# Function to create an optimized CNN model
def create_optimized_cnn(input_shape, l1_lambda=0.0001, dropout_rate=0.5):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape,
               kernel_regularizer=l1(l1_lambda), bias_regularizer=l1(l1_lambda)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        Conv1D(64, kernel_size=3, activation="relu",
               kernel_regularizer=l1(l1_lambda), bias_regularizer=l1(l1_lambda)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(64, activation="relu",
              kernel_regularizer=l1(l1_lambda), bias_regularizer=l1(l1_lambda)),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model