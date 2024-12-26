from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel
import numpy as np
import joblib


class RFModel(BaseModel):
    """
    Random Forest Classifier model.
    """

    def __init__(self) -> None:
        """
        Initializes the RFModel.
        """
        super().__init__()

    def build(self, n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
        """
        Builds the Random Forest model.

        Args:
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.

        Returns:
            RandomForestClassifier: Initialized Random Forest model.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        return self.model

    def train(self, X_train: np.ndarray, y_train) -> RandomForestClassifier:
        """
        Trains the Random Forest model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.

        Returns:
            RandomForestClassifier: Trained Random Forest model.
        """
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_2d, y_train)
        return self.model

    def predict(self, X_test) -> np.ndarray:
        """
        Generates prediction probabilities for the test data.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Probability of the positive class.
        """
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict_proba(X_test_2d)[:, 1]

    def save(self, path: str) -> None:
        """
        Saves the Random Forest model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)

    def load(self, path: str) -> RandomForestClassifier:
        """
        Loads the Random Forest model from the specified path.

        Args:
            path (str): Path to load the model from.

        Returns:
            RandomForestClassifier: Loaded Random Forest model.
        """
        self.model = joblib.load(path)
        return self.model
