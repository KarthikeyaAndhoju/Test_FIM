import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        # Load the data
        data = pd.read_csv(self.filepath)

        # Split data into features (X) and target (y)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
