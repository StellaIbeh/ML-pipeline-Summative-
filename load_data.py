# Function to load and inspect the data
def load_and_inspect_data(filepath):
    data = pd.read_csv(filepath)
    print(data.head(10))
    print(data.describe())
    print(f"Data shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    return data