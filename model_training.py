from data_preprocessing import DataPreprocessor
from sklearn.linear_model import LinearRegression
import joblib

class ModelTrainer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = LinearRegression()
        self.preprocessor = DataPreprocessor(filepath)

    def train_model(self):
        # Use DataPreprocessor to load and preprocess data
        X_train, X_test, y_train, y_test = self.preprocessor.load_and_preprocess_data()
        
        print(f"Model trained with score: {score:.2f}")

    def save_model(self, filename):
        

if __name__ == "__main__":
    filepath = "simple_dataset.csv"
    trainer = ModelTrainer(filepath)
    trainer.train_model()
    trainer.save_model("trained_model.pkl")
