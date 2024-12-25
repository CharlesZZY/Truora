import os
from typing import Tuple
from config import Config
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


class DataTransformer:
    """
    用于完成以下功能：
    1. 读取并加载数据（包括原始数据与增强数据）
    2. 对音频数据进行特征提取（Mel-Spectrogram, MFCC, Chroma等）
    3. 保存与加载NPZ格式的数据
    """

    def __init__(self, config: Config):
        self.config = config

    def save_data(features, labels, file_path):
        np.savez(file_path, features=features, labels=labels)
        print(f"Dataset saved to: {file_path}")

    def load_data_from_npz(self, file_path):
        data = np.load(file_path)
        return data

    def load_data(self, dataset_path: str, labels_df: pd.DataFrame, augmented: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        filenames = self._get_filenames(labels_df, augmented)
        features, labels = [], []

        for filename in tqdm(filenames, desc="Loading data", ncols=100, unit="file"):
            file_path = os.path.join(dataset_path, filename)
            audio_features = self.extract_features(file_path)

            story_type = self._get_story_type(filename, labels_df, augmented)

            features.append(audio_features)
            labels.append(story_type)

        return {
            "features": np.array(features),
            "labels": np.array(labels),
        }

    def _get_filenames(self, labels_df: pd.DataFrame, augmented: bool) -> list:
        if augmented:
            augmented_labels_df = pd.read_csv(self.config.augmented_label_path)
            return augmented_labels_df["filename"].tolist()
        else:
            return labels_df["filename"].tolist()

    def _get_story_type(self, filename: str, labels_df: pd.DataFrame, augmented: bool) -> str:
        if augmented:
            augmented_labels_df = pd.read_csv(self.config.augmented_label_path)
            return augmented_labels_df[augmented_labels_df["filename"] == filename]["Story_type"].values[0]
        else:
            return labels_df[labels_df["filename"] == filename]["Story_type"].values[0]

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
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
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
