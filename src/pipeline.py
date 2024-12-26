from config import Config
from utils.data_transformer import DataTransformer
from utils.evaluator import Evaluator
from utils.predictor import Predictor
from typing import Dict, Any, List, Tuple
from model import BaseModel, NNModel, RFModel, KNNModel, LRModel, SVM
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Pipeline:
    """
    Pipeline for managing data preprocessing, model training, evaluation, and prediction.

    Attributes:
        config (Config): Configuration object.
        data_transformer (DataTransformer): DataTransformer instance.
        evaluator (Evaluator): Evaluator instance.
        predictor (Predictor): Predictor instance.
        models_dict (Dict[str, BaseModel]): Dictionary of models.
        label_encoder (LabelEncoder): LabelEncoder instance.
    """

    config: Config
    data_transformer: DataTransformer
    evaluator: Evaluator
    predictor: Predictor
    models_dict: Dict[str, BaseModel]
    label_encoder: LabelEncoder

    def __init__(
        self,
        config: Config,
        data_transformer: DataTransformer,
        evaluator: Evaluator,
        predictor: Predictor,
        model_list: Dict[str, BaseModel],
    ) -> None:
        """
        Initializes the Pipeline with necessary components.

        Args:
            config (Config): Configuration object.
            data_transformer (DataTransformer): DataTransformer instance.
            evaluator (Evaluator): Evaluator instance.
            predictor (Predictor): Predictor instance.
            model_list (Dict[str, BaseModel]): Dictionary of models.
        """
        self.config = config
        self.data_transformer = data_transformer
        self.evaluator = evaluator
        self.predictor = predictor
        self.models_dict = model_list if model_list else {
            "nn_model": NNModel(),
            "rf_model": RFModel(),
            "knn_model": KNNModel(),
            "lr_model": LRModel(),
            "svm_model": SVM(),
        }
        self.init_label_encoder(self.config.labels)

    def set_model(self, name: str, model: BaseModel) -> None:
        """
        Sets or updates a model in the models dictionary.

        Args:
            name (str): Name of the model.
            model (BaseModel): Model instance.
        """
        self.models_dict[name] = model

    def init_label_encoder(self, labels: List[str]) -> None:
        """
        Initializes the LabelEncoder with the provided labels.

        Args:
            labels (List[str]): List of label names.
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.labels = self.label_encoder.classes_

    def data_preprocessing(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Performs data preprocessing by loading datasets, encoding labels, and reshaping features.

        Returns:
            List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
                Tuple containing preprocessed training, validation, and test sets.
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_transformer.get_datasets()
        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)

        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded

    def train_model(
        self,
        model_name: str,
        model: BaseModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> BaseModel:
        """
        Trains a specified model with the provided training data.

        Args:
            model_name (str): Name of the model.
            model (BaseModel): Model instance.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseModel: Trained model.
        """
        model.train(X_train, y_train, *args, **kwargs)
        self.set_model(model_name, model)
        return model

    def load_model(self, model_name: str, model_path: str) -> BaseModel:
        """
        Loads a specified model from the given path.

        Args:
            model_name (str): Name of the model.
            model_path (str): Path to load the model from.

        Returns:
            BaseModel: Loaded model.
        """
        model = self.models_dict[model_name]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified model path {model_path} does not exist.")

        print(f"Loading model: {model_name} from {model_path}")
        model.load(model_path)
        print(f"Model {model_name} successfully loaded.")

        self.set_model(model_name, model)
        return model

    def enable_model(self, model_names: List[str]) -> Dict[str, BaseModel]:
        """
        Enables specified models and disables others.

        Args:
            model_names (List[str]): List of model names to enable.

        Returns:
            Dict[str, BaseModel]: Dictionary of enabled models.
        """
        for name, model in self.models_dict.items():
            if name in model_names:
                model.enable()
            else:
                model.disable()

        enabled_models = {
            name: model for name, model in self.models_dict.items() if model.get() is not None
        }

        return enabled_models

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, test_models: List[str]) -> pd.DataFrame:
        """
        Evaluates the specified models on the test data.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
            test_models (List[str]): List of model names to evaluate.

        Returns:
            pd.DataFrame: DataFrame containing true labels, predicted labels, and predicted probabilities.
        """
        enabled_models = self.enable_model(test_models)

        if not enabled_models:
            raise ValueError("No enabled models available for evaluation, please enable at least one model.")

        print(f"Enabled models for evaluation: {list(enabled_models.keys())}")

        final_pred, final_pred_prob = self.evaluator.evaluate_ensemble_model(
            X_test=X_test,
            y_true=y_test,
            labels=self.config.labels,
            **enabled_models
        )

        predicted_labels = self.label_encoder.inverse_transform(final_pred)
        true_labels = self.label_encoder.inverse_transform(y_test)

        results = {
            "True Labels": true_labels,
            "Predicted Labels": predicted_labels,
            "Predicted Probabilities": final_pred_prob,
        }

        results_df = pd.DataFrame(results)

        print("\nEnsemble Model Evaluation Results:")
        return results_df

    def predict_audio(self, audio_path: str, language: str, use_models: List[str]) -> str:
        """
        Predicts the label for a single audio file using the specified models.

        Args:
            audio_path (str): Path to the audio file.
            language (str): Language of the audio ("English" or others).
            use_models (List[str]): List of model names to use for prediction.

        Returns:
            str: Predicted label.
        """
        enabled_models = self.enable_model(use_models)
        audio_features = self.data_transformer.extract_features(audio_path)
        language_feature = np.ones((1, audio_features.shape[1])) if language == "English" else np.zeros((1, audio_features.shape[1]))
        audio_features = np.vstack([audio_features, language_feature])
        audio_features = audio_features[np.newaxis, ..., np.newaxis]

        final_pred, final_pred_prob = self.predictor.ensemble_predict(X_test=audio_features, **enabled_models)
        predicted_label = self.label_encoder.inverse_transform(final_pred)[0]

        print(f"Predicted label: {predicted_label}")
        print(f"Predicted probabilities: {final_pred_prob}")
        return predicted_label
