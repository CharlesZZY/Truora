from typing import Any, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from .predictor import Predictor
from .plotter import plot_confusion_matrix, plot_roc_curve


class Evaluator:
    """
    Evaluates models, including metric calculations, confusion matrix, and ROC curve plotting.
    """
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Prints classification report, confusion matrix, and various metrics.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        report: str = classification_report(y_true, y_pred)
        print("Classification Report:\n", report)

        cm: np.ndarray = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)

        TN, FP, FN, TP = cm.ravel()

        sensitivity: float = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity: float = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision: float = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score: float = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        print(f"Sensitivity: {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1 Score: {f1_score:.2f}")

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

    def evaluate_ensemble_model(
        self,
        *,
        X_test: np.ndarray,
        y_true: np.ndarray,
        labels: List[str],
        **kwargs: dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the ensemble model by performing ensemble predictions and plotting evaluation metrics.

        Args:
            X_test (np.ndarray): Test features.
            y_true (np.ndarray): True labels.
            labels (List[str]): List of label names.
            **kwargs: Models to be used for prediction, passed as keyword arguments.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Final predicted classes and their probabilities.
        """
        final_pred, final_pred_prob = Predictor.ensemble_predict(X_test=X_test, **kwargs)
        print("\nEnsemble Model Evaluation:")
        self.evaluate_model(y_true, final_pred)
        plot_confusion_matrix(y_true, final_pred, labels=labels)
        plot_roc_curve(y_true, final_pred_prob)
        return final_pred, final_pred_probs
