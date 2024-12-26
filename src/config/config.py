from typing import List

class Config:
    """
    Configuration class for setting up dataset paths, labels, model parameters, and training configurations.

    Attributes:
        train_dataset (str): Path to the training dataset.
        validation_dataset (str): Path to the validation dataset.
        test_dataset (str): Path to the test dataset.
        labels (List[str]): List of label names.
        model_path (str): Directory path to save and load models.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
    """

    train_dataset: str
    validation_dataset: str 
    test_dataset: str
    labels: List[str]
    model_path: str
    epochs: int
    batch_size: int

    def __init__(
        self,
        train_dataset: str = "../datasets/train.npz",
        validation_dataset: str = "../datasets/validation.npz", 
        test_dataset: str = "../datasets/test.npz",
        labels: List[str] = ["True Story", "Deceptive Story"],
        model_path: str = "../models/",
        epochs: int = 100,
        batch_size: int = 10,
    ):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.labels = labels
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size