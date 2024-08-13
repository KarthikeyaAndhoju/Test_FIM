from data_preprocessing import load_and_preprocess_data
from sklearn.linear_model import LinearRegression
import pandas as pd

class ModelTrainer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = LinearRegression()

    def train_model(self):
        # Load and preprocess the data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(self.filepath)
        print(f"Model trained with score: {score:.2f}")

    def save_model(self, filename):
        import joblib

if __name__ == "__main__":
    filepath = "simple_dataset.csv"
    trainer = ModelTrainer(filepath)
    trainer.train_model()
    trainer.save_model("trained_model.pkl")
