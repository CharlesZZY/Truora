import numpy as np
from typing import Tuple


class Predictor:
    """
    用于模型预测及输出结果。包含集成预测的方法。
    """
    @staticmethod
    def ensemble_predict(
        X_test: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        model_list = ["nn_models", "rf_model", "knn_model", "lr_model", "svm_model"]
        models = {k: v for k, v in kwargs.items() if k in model_list}

        if not any(models.values()):
            raise ValueError("At least one model is required!")

        predictions = []

        for model_name, model in models.items():
            if model is None or not model.use:
                continue

            if isinstance(model, list):
                model_preds = np.mean([m.predict(X_test) for m in model], axis=0)
                predictions.append(model_preds)
                print(f"{model_name} Predictions: {model_preds}")
            else:
                preds = model.predict(X_test)
                predictions.append(preds)
                print(f"{model_name} Predictions: {preds}")

        num_sources = len(predictions)
        weights = kwargs.get("weights")
        if weights is None:
            weights = [1.0 / num_sources] * num_sources
        elif len(weights) != num_sources:
            raise ValueError("The number of weights must match the number of models used for prediction!")

        final_pred_prob = np.average(predictions, axis=0, weights=weights)
        final_pred = (final_pred_prob > 0.5).astype(int)

        return final_pred, final_pred_prob
