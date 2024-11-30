import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

def evaluate_model(model_path, X_test, y_test):
    """
    Loads the trained model and evaluates it on test data.
    
    Parameters:
        model_path (str): Path to the saved model file.
        X_test (numpy array): Test features.
        y_test (numpy array): Test labels.
    """
    # Load the saved model
    model = load_model(model_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Diabetes Absent', 'Diabetes Present'],
                yticklabels=['Diabetes Absent', 'Diabetes Present'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=['Diabetes Absent', 'Diabetes Present']))
