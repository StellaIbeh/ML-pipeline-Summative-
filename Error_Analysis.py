from scikit-learn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Function for error analysis
def error_analysis(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int).reshape(-1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()