from typing import Tuple
from model import NNModel, RFModel, KNNModel, LRModel, SVM


class ModelBuilder:
    """
    Responsible for constructing and returning the required models.
    Currently includes CNN-LSTM model (with Attention) and various traditional machine learning models.
    """

    @staticmethod
    def build_nn_model(input_shape: Tuple[int, int, int]) -> NNModel:
        """
        Builds the Neural Network model.

        Args:
            input_shape (Tuple[int, int, int]): Shape of the input data.

        Returns:
            NNModel: Built Neural Network model.
        """
        nn_model: NNModel = NNModel()
        nn_model.build(input_shape)
        return nn_model

    @staticmethod
    def build_rf_model(random_state: int = 42, n_estimators: int = 100) -> RFModel:
        """
        Builds the Random Forest model.

        Args:
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.

        Returns:
            RFModel: Built Random Forest model.
        """
        rf_model: RFModel = RFModel()
        rf_model.build(random_state=random_state, n_estimators=n_estimators)
        return rf_model

    @staticmethod
    def build_knn_model(n_neighbors: int = 3) -> KNNModel:
        """
        Builds the K-Nearest Neighbors model.

        Args:
            n_neighbors (int, optional): Number of neighbors to use. Defaults to 3.

        Returns:
            KNNModel: Built KNN model.
        """
        knn_model: KNNModel = KNNModel()
        knn_model.build(n_neighbors=n_neighbors)
        return knn_model

    @staticmethod
    def build_logistic_regression_model(C: float = 1, random_state: int = 42, max_iter: int = 100) -> LRModel:
        """
        Builds the Logistic Regression model.

        Args:
            C (float, optional): Inverse of regularization strength. Defaults to 1.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            LRModel: Built Logistic Regression model.
        """
        lr_model: LRModel = LRModel()
        lr_model.build(C=C, random_state=random_state, max_iter=max_iter)
        return lr_model

    @staticmethod
    def build_svm_model(C: float = 1.0, kernel: str = "rbf") -> SVM:
        """
        Builds the Support Vector Machine model.

        Args:
            C (float, optional): Regularization parameter. Defaults to 1.0.
            kernel (str, optional): Kernel type to be used in the algorithm. Defaults to 'rbf'.

        Returns:
            SVM: Built SVM model.
        """
        svm_model: SVM = SVM()
        svm_model.build(C=C, kernel=kernel)
        return svm_model
