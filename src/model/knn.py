from sklearn.neighbors import KNeighborsClassifier
from .base import BaseModel
import numpy as np
import joblib


class KNNModel(BaseModel):
    """
    K-Nearest Neighbors Classifier model.
    """

    def __init__(self) -> None:
        """
        Initializes the KNNModel.
        """
        super().__init__()

    def build(self, n_neighbors: int = 5) -> KNeighborsClassifier:
        """
        Builds the KNN model.

        Args:
            n_neighbors (int, optional): Number of neighbors to use. Defaults to 5.

        Returns:
            KNeighborsClassifier: Initialized KNN model.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
        """
        Trains the KNN model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.

        Returns:
            KNeighborsClassifier: Trained KNN model.
        """
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_2d, y_train)
        return self.model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the test data.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test_2d)

    def save(self, path: str) -> None:
        """
        Saves the KNN model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)

    def load(self, path: str) -> KNeighborsClassifier:
        """
        Loads the KNN model from the specified path.

        Args:
            path (str): Path to load the model from.

        Returns:
            KNeighborsClassifier: Loaded KNN model.
        """
        self.model = joblib.load(path)
        return self.model
