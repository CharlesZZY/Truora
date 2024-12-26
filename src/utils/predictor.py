from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Predictor:
    """
    Used for model prediction and output results. Includes methods for ensemble predictions.
    """
    @staticmethod
    def ensemble_predict(
        X_test: np.ndarray,
        **kwargs: dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs ensemble prediction by averaging the probabilities from multiple models.

        Args:
            X_test (np.ndarray): Test features.
            **kwargs: Models to be used for prediction, passed as keyword arguments.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Final predicted classes and their probabilities.
        """
        model_list: List[str] = ["nn_model", "rf_model", "knn_model", "lr_model", "svm_model"]
        models: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in model_list}

        if not any(models.values()):
            raise ValueError("At least one model is required!")

        predictions: List[np.ndarray] = []

        for model_name, model in models.items():
            if model is None or not model.use:
                continue
            preds: np.ndarray = model.predict(X_test)
            predictions.append(preds)
            print(f"{model_name} Predictions: {preds}")

        num_sources: int = len(predictions)
        weights: Optional[List[float]] = kwargs.get("weights")
        if weights is None:
            weights = [1.0 / num_sources] * num_sources
        elif len(weights) != num_sources:
            raise ValueError("The number of weights must match the number of models used for prediction!")

        final_pred_prob: np.ndarray = np.average(predictions, axis=0, weights=weights)
        final_pred: np.ndarray = (final_pred_prob > 0.5).astype(int)

        return final_pred, final_pred_prob
