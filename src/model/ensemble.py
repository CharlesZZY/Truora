from .base import BaseModel
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


class EnsembleModel(BaseModel):
    """
    Ensemble model that classifies by weighted averaging the prediction probabilities of five base models.
    The weights are automatically adjusted using training data to minimize prediction error.
    """

    def __init__(self, models: Dict[str, BaseModel]) -> None:
        """
        Initialize the ensemble model.

        Args:
            models (Dict[str, BaseModel]): A dictionary containing five base models.
        """
        super().__init__()
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # Initial equal weights
        self.weight_model = LogisticRegression()  # Meta-classifier for learning weights

    def build(self) -> None:
        """
        Build the ensemble model (no actual model structure needs to be built).
        """
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs: Any) -> None:
        """
        Train the ensemble model by fitting a logistic regression model to learn the weights.

        Args:
            X_train (np.ndarray): Training features, shape (n_samples, 167, 7500, 1).
            y_train (np.ndarray): Training labels, shape (n_samples,).
            X_val (np.ndarray, optional): Validation features. Defaults to None.
            y_val (np.ndarray, optional): Validation labels. Defaults to None.
        """
        predictions = []
        for _, model in self.models.items():
            preds = model.predict(X_train)
            predictions.append(preds.reshape(-1, 1))

        # Combine all model prediction probabilities as new features
        ensemble_features = np.hstack(predictions)  # Shape (n_samples, 5)

        self.weight_model.fit(ensemble_features, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble model.

        Args:
            X_test (np.ndarray): Test features, shape (n_samples, 167, 7500, 1).

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples,)
        """
        predictions = []
        for _, model in self.models.items():
            preds = model.predict(X_test)
            predictions.append(preds.reshape(-1, 1))

        ensemble_features = np.hstack(predictions)  # Shape (n_samples, 5)
        final_pred_prob = self.weight_model.predict_proba(ensemble_features)[:, 1]
        return final_pred_prob

    def save(self, path: str) -> None:
        """
        Save the weights of the ensemble model.

        Args:
            path (str): Save path.
        """
        joblib.dump(self.weight_model, path)

    def load(self, path: str) -> None:
        """
        Load the weights of the ensemble model.

        Args:
            path (str): Load path.
        """
        self.weight_model = joblib.load(path)
