import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l1
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
# Main pipeline execution
if __name__ == "__main__":
    # Load and preprocess the data
    filepath = r'C:\Users\HP\ML-pipeline-Summative-\diabetes.csv'
    data = load_and_inspect_data(filepath)
    visualize_data(data)
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train and evaluate the simple CNN model
    simple_model = create_simple_cnn(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_model(simple_model, X_train, y_train, X_test, y_test)
    evaluate_model(simple_model, X_test, y_test)
    
    # Train and evaluate the optimized CNN model
    optimized_model = create_optimized_cnn(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_model(optimized_model, X_train, y_train, X_test, y_test)
    evaluate_model(optimized_model, X_test, y_test)
    
    # Perform error analysis
    error_analysis(optimized_model, X_test, y_test)