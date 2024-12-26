from config import Config
import librosa
from typing import Tuple
import numpy as np


class DataTransformer:
    """
    Handles data loading and feature extraction for audio datasets.

    Attributes:
        config (Config): Configuration object containing dataset paths and parameters.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the DataTransformer with the given configuration.

        Args:
            config (Config): Configuration object.
        """
        self.config: Config = config

    @staticmethod
    def load_data_from_npz(file_path: str) -> np.ndarray:
        """
        Loads data from a .npz file.

        Args:
            file_path (str): Path to the .npz file.

        Returns:
            np.ndarray: Loaded data.
        """
        data = np.load(file_path)
        return data

    def get_datasets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves training, validation, and test datasets.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Tuple containing features and labels for training, validation, and test sets.
        """
        train_data = self.load_data_from_npz(self.config.train_dataset)
        validation_data = self.load_data_from_npz(self.config.validation_dataset)
        test_data = self.load_data_from_npz(self.config.test_dataset)
        return train_data["features"], train_data["labels"], validation_data["features"], validation_data["labels"], test_data["features"], test_data["labels"]

    @staticmethod
    def extract_features(
        file_path: str,
        sr: int = 16000,
        n_mels: int = 128,
        duration: int = 240,
    ) -> np.ndarray:
        """
        Extracts audio features from a given file, including Mel-Spectrogram, MFCC, Chroma, ZCR, energy, duration,
        spectral centroid, spectral rolloff, and spectral flux. All features are concatenated into a single matrix
        with fixed length processing.

        Args:
            file_path (str): Path to the audio file.
            sr (int, optional): Sampling rate. Defaults to 16000.
            n_mels (int, optional): Number of Mel bands. Defaults to 128.
            duration (int, optional): Duration of the audio in seconds. Defaults to 240.

        Returns:
            np.ndarray: Extracted and concatenated features.
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


def split_and_save_data(
    original_data_path: str,
    augmented_data_path: str,
    save_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> None:
    """
    Splits the original and augmented data into training, validation, and test sets and saves them.

    Args:
        original_data_path (str): Path to the original data .npz file.
        augmented_data_path (str): Path to the augmented data .npz file.
        save_path (str): Directory path to save the split datasets.
        train_ratio (float, optional): Ratio of data to be used for training. Defaults to 0.7.
        val_ratio (float, optional): Ratio of data to be used for validation. Defaults to 0.15.
    """
    # Validate train_ratio and val_ratio
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("The sum of train_ratio and val_ratio must be less than 1")

    original_data: np.lib.npyio.NpzFile = np.load(original_data_path)
    augmented_data: np.lib.npyio.NpzFile = np.load(augmented_data_path)

    original_features = original_data['features']
    original_labels = original_data['labels']
    augmented_features = augmented_data['features']
    augmented_labels = augmented_data['labels']

    original_train_size = int(len(original_features) * train_ratio)
    augmented_train_size = int(len(augmented_features) * train_ratio)

    original_val_size = int(len(original_features) * val_ratio)
    augmented_val_size = int(len(augmented_features) * val_ratio)

    original_train_features = original_features[:original_train_size]
    original_train_labels = original_labels[:original_train_size]
    augmented_train_features = augmented_features[:augmented_train_size]
    augmented_train_labels = augmented_labels[:augmented_train_size]

    original_val_features = original_features[original_train_size:original_train_size + original_val_size]
    original_val_labels = original_labels[original_train_size:original_train_size + original_val_size]
    augmented_val_features = augmented_features[augmented_train_size:augmented_train_size + augmented_val_size]
    augmented_val_labels = augmented_labels[augmented_train_size:augmented_train_size + augmented_val_size]

    original_test_features = original_features[original_train_size + original_val_size:]
    original_test_labels = original_labels[original_train_size + original_val_size:]
    augmented_test_features = augmented_features[augmented_train_size + augmented_val_size:]
    augmented_test_labels = augmented_labels[augmented_train_size + augmented_val_size:]

    train_features = np.concatenate((original_train_features, augmented_train_features), axis=0)
    train_labels = np.concatenate((original_train_labels, augmented_train_labels), axis=0)

    validation_features = np.concatenate((original_val_features, augmented_val_features), axis=0)
    validation_labels = np.concatenate((original_val_labels, augmented_val_labels), axis=0)

    test_features = np.concatenate((original_test_features, augmented_test_features), axis=0)
    test_labels = np.concatenate((original_test_labels, augmented_test_labels), axis=0)

    np.savez(f"{save_path}train.npz", features=train_features, labels=train_labels)
    np.savez(f"{save_path}validation.npz", features=validation_features, labels=validation_labels)
    np.savez(f"{save_path}test.npz", features=test_features, labels=test_labels)
