import os
import numpy as np
import pandas as pd
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import *
from keras import models, layers, callbacks
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from keras.models import load_model
os.environ["KERAS_BACKEND"] = "torch"


class Config:
  def __init__(
      self,
      dataset_path: str = "../datasets/",
      story_path: str = "../datasets/CBU0521DD_stories/",
      augmented_story_path: str = "../datasets/CBU0521DD_stories/_augmented/",
      label_path: str = "../datasets/CBU0521DD_stories_attributes.csv",
      augmented_label_path: str = "../datasets/CBU0521DD_stories_attributes_augmented.csv",
      model_path: str = "../models/",
      epoch: int = 100,
      batch_size: int = 10,
  ):
    self.dataset_path = dataset_path
    self.story_path = story_path
    self.augmented_story_path = augmented_story_path
    self.label_path = label_path
    self.augmented_label_path = augmented_label_path
    self.model_path = model_path
    self.epoch = epoch
    self.batch_size = batch_size


config = Config(
    dataset_path="../datasets/",
    story_path="../datasets/CBU0521DD_stories/",
    augmented_story_path="../datasets/CBU0521DD_stories/_augmented/",
    label_path="../datasets/CBU0521DD_stories_attributes.csv",
    augmented_label_path="../datasets/CBU0521DD_stories_attributes_augmented.csv",
    model_path="../models/",
    epoch=100,
    batch_size=10,
)


class DataTransformer:
  """
  用于完成以下功能：
  1. 读取并加载数据（包括原始数据与增强数据）
  2. 对音频数据进行特征提取（Mel-Spectrogram, MFCC, Chroma等）
  3. 保存与加载NPZ格式的数据
  """

  def __init__(self, config: Config):
    self.config = config

  def load_data(
      self,
      dataset_path: str,
      labels_df: pd.DataFrame,
      augmented: bool = False
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    augmented=True时，读取增强后的数据及标签，否则读取原始数据。
    返回 features, labels
    """
    features = []
    labels = []

    if augmented:
      augmented_labels_df = pd.read_csv(self.config.augmented_label_path)
      filenames = augmented_labels_df["filename"].tolist()
    else:
      filenames = labels_df["filename"].tolist()

    for filename in tqdm(filenames, desc="Loading data", ncols=100, unit="file"):
      file_path = os.path.join(dataset_path, filename)
      audio_features = self.extract_features(file_path)

      if augmented:
        story_type = augmented_labels_df[
            augmented_labels_df["filename"] == filename
        ]["Story_type"].values[0]
      else:
        story_type = labels_df[
            labels_df["filename"] == filename
        ]["Story_type"].values[0]

      features.append(audio_features)
      labels.append(story_type)

    return np.array(features), np.array(labels)

  @staticmethod
  def extract_features(
      file_path: str,
      sr: int = 16000,
      n_mels: int = 128,
      duration: int = 240,
  ) -> np.ndarray:
    """
    提取音频特征，包括Mel-Spectrogram、MFCC、Chroma、ZCR、能量、持续时间、谱质心、滚降点等。
    最终将所有特征拼接到同一矩阵中，并做定长处理。
    """
    audio, sr = librosa.load(file_path, sr=sr)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

    short_term_energy = np.sum(audio ** 2) / len(audio)

    duration_feature = len(audio) / sr

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, roll_percent=0.85
    )
    spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)

    features = np.vstack([
        log_mel_spectrogram,
        mfcc,
        chroma,
        zero_crossing_rate,
        spectral_centroid,
        spectral_rolloff,
        spectral_flux,
    ])

    target_length = int(sr * duration / 512)
    if features.shape[1] < target_length:
      padding = np.zeros((features.shape[0], target_length - features.shape[1]))
      features = np.concatenate([features, padding], axis=1)
    else:
      features = features[:, :target_length]

    additional_features = np.array([short_term_energy, duration_feature])
    additional_features = np.repeat(additional_features[:, np.newaxis], features.shape[1], axis=1)
    features = np.concatenate([features, additional_features], axis=0)

    return features

  @staticmethod
  def save_data(features: np.ndarray, labels: np.ndarray, file_path: str) -> None:
    np.savez(file_path, features=features, labels=labels)
    print(f"Dataset saved to: {file_path}")

  @staticmethod
  def load_data_from_npz(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(file_path)
    return data["features"], data["labels"]


class ModelBuilder:
  """
  负责构建与返回所需模型。
  目前包含：CNN-LSTM模型（带Attention）以及随机森林模型。
  """

  def build_cnn_lstm_model(self, input_shape: Tuple[int, int, int]) -> models.Model:
    model_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Reshape((1, 128))(x)  # (batch_size, sequence_length=1, feature_dim=128)
    attention_output = layers.Attention()([x, x])  # Self-Attention

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(attention_output)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer="l2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=model_input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

  @staticmethod
  def build_rf_model(random_state: int = 42, n_estimators: int = 100) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


class Trainer:
  """
  负责模型的训练与可视化（如Loss、Accuracy曲线）。
  """

  def __init__(self, config: Config, model_builder: ModelBuilder):
    self.config = config
    self.model_builder = model_builder
    self.models_list: List[models.Model] = []
    self.rf_model: Optional[RandomForestClassifier] = None

  @staticmethod
  def show_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

  def train_cnn_lstm_ensemble(
      self,
      X_train: np.ndarray,
      y_train: np.ndarray,
      X_val: np.ndarray,
      y_val: np.ndarray,
      n_models: int = 3,
      continue_training: bool = False
  ) -> List[models.Model]:
    """
    训练多个CNN-LSTM模型（集成），或加载已有模型继续训练。
    """
    if continue_training:
      for i in range(n_models):
        model_path = os.path.join(self.config.model_path, f"best_model_{i + 1}.keras")
        loaded_model = load_model(model_path)
        self.models_list.append(loaded_model)
    else:
      self.models_list = []

    for i in range(n_models):
      if continue_training:
        model = self.models_list[i]
        print(f"继续训练Model {i+1}")
      else:
        print(f"训练Model {i+1}")
        model = self.model_builder.build_cnn_lstm_model(
            (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        )

      early_stopping = callbacks.EarlyStopping(monitor="loss", patience=100)
      model_checkpoint = callbacks.ModelCheckpoint(
          filepath=os.path.join(self.config.model_path, f"best_model_{i + 1}.keras"),
          monitor="val_accuracy",
          save_best_only=True
      )
      callbacks_list = [early_stopping, model_checkpoint]

      history = model.fit(
          X_train,
          y_train,
          epochs=self.config.epoch,
          batch_size=self.config.batch_size,
          validation_data=(X_val, y_val),
          callbacks=callbacks_list
      )
      self.show_history(history)
      self.models_list.append(model)

    return self.models_list

  def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    训练随机森林模型。
    """
    self.rf_model = self.model_builder.build_rf_model()
    # 对高维特征进行降维处理，展开为2D
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    self.rf_model.fit(X_train_2d, y_train)
    return self.rf_model


class Predictor:
  """
  用于模型预测及输出结果。包含集成预测的方法。
  """
  @staticmethod
  def ensemble_predict(
      models_list: List[models.Model],
      rf_model: RandomForestClassifier,
      X_test: np.ndarray,
      weights: Optional[List[float]] = None
  ) -> np.ndarray:
    nn_preds = np.zeros((len(models_list), X_test.shape[0]))
    for i, model in enumerate(models_list):
      nn_preds[i] = model.predict(X_test).flatten()

    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    rf_preds = rf_model.predict_proba(X_test_2d)[:, 1]

    if weights is None:
      weights = [1.0 / len(models_list)] * len(models_list)

    nn_pred_avg = np.average(nn_preds, axis=0, weights=weights)

    final_pred_prob = (nn_pred_avg + rf_preds) / 2.0
    final_pred = (final_pred_prob > 0.5).astype(int)
    return final_pred, final_pred_prob


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

    sensitivity = TP / (TP + FN)  # TPR
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")

  @staticmethod
  def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

  @staticmethod
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

  def evaluate_ensemble_model(
      self,
      models_list: List[models.Model],
      rf_model: RandomForestClassifier,
      X_test: np.ndarray,
      y_true: np.ndarray,
      labels: List[str]
  ) -> Tuple[np.ndarray, np.ndarray]:
    final_pred, final_pred_prob = Predictor.ensemble_predict(models_list, rf_model, X_test)
    print("\nEnsemble Model Evaluation:")
    self.evaluate_model(y_true, final_pred)
    self.plot_confusion_matrix(y_true, final_pred, labels=labels)
    self.plot_roc_curve(y_true, final_pred_prob)
    return final_pred, final_pred_prob


class Pipeline:
  """
  整合所有步骤：数据加载、特征处理、模型训练、预测以及评估。
  """

  def __init__(self, config: Config):
    self.config = config
    self.data_transformer = DataTransformer(config)
    self.model_builder = ModelBuilder()
    self.trainer = Trainer(config, self.model_builder)
    self.evaluator = Evaluator()
    self.label_encoder = LabelEncoder()
    self.X_train = self.X_val = self.X_test = None
    self.y_train = self.y_val = self.y_test = None

  def run(self):
    # 1. 读取标签
    labels_df = pd.read_csv(self.config.label_path)

    # 2. 加载原始数据与增强数据
    original_features, original_labels = self.data_transformer.load_data(
        self.config.story_path, labels_df, augmented=False
    )
    augmented_features, augmented_labels = self.data_transformer.load_data(
        self.config.augmented_story_path, labels_df, augmented=True
    )

    # 3. 合并
    features = np.concatenate((original_features, augmented_features), axis=0)
    labels = np.concatenate((original_labels, augmented_labels), axis=0)
    print(
        f"Original dataset size: {original_features.shape[0]}, "
        f"Augmented dataset size: {augmented_features.shape[0]}, "
        f"Combined dataset size: {features.shape[0]}"
    )

    # 4. 保存数据（可选）
    self.data_transformer.save_data(
        original_features,
        original_labels,
        os.path.join(self.config.dataset_path, "original_data.npz"),
    )
    self.data_transformer.save_data(
        augmented_features,
        augmented_labels,
        os.path.join(self.config.dataset_path, "augmented_data.npz"),
    )
    self.data_transformer.save_data(
        features, labels, os.path.join(self.config.dataset_path, "data.npz")
    )

    # 5. 标签数值编码
    labels_encoded = self.label_encoder.fit_transform(labels)
    print("Encoded labels:", labels_encoded)

    # 6. 划分训练集、验证集、测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels_encoded, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 增加通道维度
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # 7. 训练多个CNN-LSTM模型（此处n_models=1方便演示，可根据需求调整）
    models_list = self.trainer.train_cnn_lstm_ensemble(
        X_train, y_train, X_val, y_val, n_models=1, continue_training=False
    )

    # 8. 训练随机森林模型
    rf_model = self.trainer.train_random_forest(X_train, y_train)

    # 9. 集成预测
    final_pred, final_pred_prob = Predictor.ensemble_predict(models_list, rf_model, X_test)

    ensemble_accuracy = accuracy_score(y_test, final_pred)
    print(f"Ensemble Model Test Accuracy: {ensemble_accuracy:.4f}")

    # 10. 评估集成模型
    predicted_labels = self.label_encoder.inverse_transform(final_pred)
    true_labels = self.label_encoder.inverse_transform(y_test)
    for true_label, pred_label in zip(true_labels, predicted_labels):
      print(f"Actual: {true_label}, Predicted: {pred_label}")

    # 可视化评估指标
    self.evaluator.evaluate_ensemble_model(
        models_list,
        rf_model,
        X_test,
        y_test,
        labels=["False Story", "True Story"]
    )

  def inference(self, audio_data: np.ndarray, sr: int = 16000) -> str:
      """
      接收原始音频数组 audio_data 进行推理，输出 "True" 或 "False"。
      self.models_list 和 self.rf_model 需预先加载或训练好。
      """
      # 1. 提取特征
      audio_features = self.data_transformer.extract_features_from_array(audio_data, sr=sr)

      # 2. 调整维度，保证符合模型输入 (batch_size, height, width, channels)
      audio_features = audio_features[..., np.newaxis]         # (height, width) -> (height, width, 1)
      audio_features = np.expand_dims(audio_features, axis=0)  # -> (1, height, width, 1)

      # 3. 使用集成模型进行预测
      final_pred, _ = Predictor.ensemble_predict(self.models_list, self.rf_model, audio_features)

      # 4. 根据二值结果输出字符
      pred_label = "True" if final_pred[0] == 1 else "False"
      return pred_label
