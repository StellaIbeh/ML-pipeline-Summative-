import matplotlib.pyplot as plt
import seaborn as sns
# Function to visualize data
def visualize_data(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.boxplot(data=data)
    plt.title("Boxplot for Outliers")
    plt.show()