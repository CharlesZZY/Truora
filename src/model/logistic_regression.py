from sklearn.linear_model import LogisticRegression
from .base import BaseModel
import numpy as np
import joblib


class LRModel(BaseModel):
    """
    Logistic Regression Classifier model.
    """

    def __init__(self) -> None:
        """
        Initializes the LRModel.
        """
        super().__init__()

    def build(self, C: float = 1.0, random_state: int = 42, max_iter: int = 100) -> LogisticRegression:
        """
        Builds the Logistic Regression model.

        Args:
            C (float, optional): Inverse of regularization strength. Defaults to 1.0.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            LogisticRegression: Initialized Logistic Regression model.
        """
        self.model = LogisticRegression(C=C, random_state=random_state, max_iter=max_iter)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Trains the Logistic Regression model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.

        Returns:
            LogisticRegression: Trained Logistic Regression model.
        """
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_2d, y_train)
        return self.model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
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
        Saves the Logistic Regression model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)

    def load(self, path: str) -> LogisticRegression:
        """
        Loads the Logistic Regression model from the specified path.

        Args:
            path (str): Path to load the model from.

        Returns:
            LogisticRegression: Loaded Logistic Regression model.
        """
        self.model = joblib.load(path)
        return self.model
