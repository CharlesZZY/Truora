from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseModel(ABC):
    """
    Abstract base class for models, supporting lifecycle management and core functionality.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseModel with default settings.
        """
        self.use: bool = True
        self.model: Optional[Any] = None

    @abstractmethod
    def build(self, **kwargs: Any) -> None:
        """
        Build the model architecture.

        Args:
            **kwargs: Additional keyword arguments for building the model.
        """
        pass

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, **kwargs: Any) -> None:
        """
        Train the model on the provided data.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            **kwargs: Additional keyword arguments for training.
        """
        pass

    @abstractmethod
    def predict(self, X_test: Any) -> Any:
        """
        Generate predictions for the provided test data.

        Args:
            X_test (Any): Test features.

        Returns:
            Any: Predictions.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.

        Args:
            path (str): File path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from the specified path.

        Args:
            path (str): File path to load the model from.
        """
        pass

    def get(self) -> Optional[Any]:
        """
        Retrieve the model instance if enabled.

        Returns:
            Optional[Any]: The model instance or None if disabled.
        """
        return self.model if self.use else None

    def enable(self) -> None:
        """
        Enable the model for use.
        """
        self.use = True

    def disable(self) -> None:
        """
        Disable the model from being used.
        """
        self.use = False
