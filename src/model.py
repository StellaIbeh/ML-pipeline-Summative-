import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

def create_optimized_model(l1_lambda=0.0001, dropout_rate=0.5):
    """
    Creates a CNN model with L1 regularization and Dropout.
    
    Parameters:
        l1_lambda (float): Regularization strength.
        dropout_rate (float): Dropout rate.

    Returns:
        optimized_model (Sequential): Compiled CNN model.
    """
    optimized_model = Sequential()

    # First Convolutional Layer
    optimized_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(9, 1),
                     kernel_regularizer=l1(l1_lambda), bias_regularizer=l1(l1_lambda)))
    optimized_model.add(MaxPooling1D(pool_size=2))
    optimized_model.add(Dropout(dropout_rate))

    # Second Convolutional Layer
    optimized_model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     kernel_regularizer=l1(l1_lambda), bias_regularizer=l1(l1_lambda)))
    optimized_model.add(Dropout(dropout_rate))

    # Fully Connected Layers
    optimized_model.add(Flatten())
    optimized_model.add(Dense(64, activation='relu',
                    kernel_regularizer=l1(l1_lambda), bias_regularizer=l1(l1_lambda)))
    optimized_model.add(Dropout(dropout_rate))

    optimized_model.add(Dense(64, activation='relu',
                    kernel_regularizer=l1(l1_lambda), bias_regularizer=l1(l1_lambda)))
    optimized_model.add(Dropout(dropout_rate))

    # Output Layer
    optimized_model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = RMSprop(learning_rate=0.01)
    optimized_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return optimized_model

def train_model(X_train, y_train, X_test, y_test, epochs=150, batch_size=32):
    """
    Trains the CNN model and saves it to disk.

    Returns:
        history (History): Training history object.
        model (Sequential): Trained model.
    """
    model = create_optimized_model()

    # Define Early Stopping
    early_stopping = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=early_stopping)

    # Save the model
    model.save('optimized_model.h5')

    return history, model

