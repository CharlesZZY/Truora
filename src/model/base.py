from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseModel(ABC):
    """
    Abstract base class for models, supporting lifecycle management and core functionality.
    """

    def __init__(self):
        self.use = True
        self.model: Optional[Any] = None

    @abstractmethod
    def build_model(self, **kwargs) -> None:
        """
        Build the model architecture.
        """
        pass

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, **kwargs) -> None:
        """
        Train the model on the provided data.
        """
        pass

    @abstractmethod
    def predict(self, X_test: Any) -> Any:
        """
        Generate predictions for the provided test data.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from the specified path.
        """
        pass

    def get(self) -> Optional[Any]:
        """
        Retrieve the model instance if enabled.
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

    def __repr__(self):
        return f"BaseModel(use={self.use}, config={self.config})"
