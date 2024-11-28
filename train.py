from keras.callbacks import EarlyStopping
# Function to train a model
def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])
    return history