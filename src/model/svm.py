from sklearn.svm import SVC
import numpy as np
import joblib
from .base import BaseModel


class SVM(BaseModel):
    """
    Support Vector Machine Classifier model.
    """

    def __init__(self) -> None:
        """
        Initializes the SVM model.
        """
        super().__init__()

    def build(self, C: float = 1.0, kernel: str = 'rbf') -> SVC:
        """
        Builds the SVM model.

        Args:
            C (float, optional): Regularization parameter. Defaults to 1.0.
            kernel (str, optional): Kernel type to be used in the algorithm. Defaults to 'rbf'.

        Returns:
            SVC: Initialized SVM model.
        """
        self.model = SVC(C=C, kernel=kernel, probability=True)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """
        Trains the SVM model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.

        Returns:
            SVC: Trained SVM model.
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
        Saves the SVM model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        joblib.dump(self.model, path)

    def load(self, path: str) -> SVC:
        """
        Loads the SVM model from the specified path.

        Args:
            path (str): Path to load the model from.

        Returns:
            SVC: Loaded SVM model.
        """
        self.model = joblib.load(path)
        return self.model
