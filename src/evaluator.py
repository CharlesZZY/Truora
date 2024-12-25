import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from .predictor import Predictor


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


class Evaluator:
    """
    对模型进行评估，包括指标计算、混淆矩阵与ROC曲线绘制等。
    """
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        report = classification_report(y_true, y_pred)
        print("Classification Report:\n", report)

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)

        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        print(f"Sensitivity: {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1 Score: {f1_score:.2f}")

    def evaluate_ensemble_model(
        self,
        *,
        X_test: np.ndarray,
        y_true: np.ndarray,
        labels: List[str],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        final_pred, final_pred_prob = Predictor.ensemble_predict(X_test=X_test, **kwargs)
        print("\nEnsemble Model Evaluation:")
        self.evaluate_model(y_true, final_pred)
        plot_confusion_matrix(y_true, final_pred, labels=labels)
        plot_roc_curve(y_true, final_pred_prob)
        return final_pred, final_pred_prob
