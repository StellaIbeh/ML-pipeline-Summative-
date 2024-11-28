from scikit-learn.model_selection import train_test_split
from scikit-learn.preprocessing import StandardScaler
# Function to preprocess data
def preprocess_data(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test