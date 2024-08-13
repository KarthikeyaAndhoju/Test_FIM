from data_preprocessing import load_and_preprocess_data
from sklearn.linear_model import LinearRegression

def train_model(filepath):
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)


if __name__ == "__main__":
    filepath = "simple_dataset.csv"  
    train_model(filepath)
