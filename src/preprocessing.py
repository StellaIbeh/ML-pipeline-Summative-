import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path, target_column='Outcome', test_size=0.2, random_state=42):
    """
    Preprocesses the dataset: scales, reshapes, and splits data for training/testing.
    
    Parameters:
        file_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target column.
        test_size (float): Fraction of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays): Processed data splits.
    """
    # Load dataset
    data = pd.read_csv(file_path)

    # Separate features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for CNN
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
