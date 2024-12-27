from typing import List, Tuple
import os
import numpy as np
import librosa
import pandas as pd

CONFIG = {
    "dataset_path": "../datasets/",
    "story_path": "../datasets/CBU5201DD_stories/",
    "label_path": "../datasets/CBU5201DD_stories_attributes.csv",
    "augmented_story_path": "../datasets/CBU5201DD_stories/_augmented/",
    "augmented_label_path": "../datasets/CBU5201DD_stories_attributes_augmented.csv",
}


def time_stretch(y: np.ndarray, rate: float = 1.2) -> np.ndarray:
    """
    Applies time stretching to the audio signal.

    Args:
        y (np.ndarray): Audio signal.
        rate (float, optional): Stretching rate. Defaults to 1.2.

    Returns:
        np.ndarray: Time-stretched audio signal.
    """
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: int = 4) -> np.ndarray:
    """
    Applies pitch shifting to the audio signal.

    Args:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        n_steps (int, optional): Number of steps to shift. Defaults to 4.

    Returns:
        np.ndarray: Pitch-shifted audio signal.
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """
    Adds Gaussian noise to the audio signal.

    Args:
        y (np.ndarray): Audio signal.
        noise_level (float, optional): Noise level. Defaults to 0.005.

    Returns:
        np.ndarray: Noisy audio signal.
    """
    noise = np.random.randn(len(y))
    return y + noise_level * noise


def change_volume(y: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Changes the volume of the audio signal.

    Args:
        y (np.ndarray): Audio signal.
        factor (float, optional): Volume change factor. Defaults to 1.2.

    Returns:
        np.ndarray: Audio signal with changed volume.
    """
    return y * factor


def process_audio(filename: str, story_path: str, sr: int = 16000) -> List[np.ndarray]:
    """
    Processes an audio file by applying various augmentation techniques.

    Args:
        filename (str): Name of the audio file.
        story_path (str): Path to the directory containing the audio file.
        sr (int, optional): Sampling rate. Defaults to 16000.

    Returns:
        List[np.ndarray]: List containing the original and augmented audio signals.
    """
    y, _ = librosa.load(os.path.join(story_path, filename), sr=sr)

    augmented_audios: List[np.ndarray] = []
    augmented_audios.append(time_stretch(y))
    augmented_audios.append(pitch_shift(y, sr))
    augmented_audios.append(add_noise(y))
    augmented_audios.append(change_volume(y))

    return [y] + augmented_audios


def save_data(features: np.ndarray, labels: np.ndarray, file_path: str) -> None:
    """
    Saves features and labels to a .npz file.

    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Label array.
        file_path (str): Path to save the .npz file.
    """
    np.savez(file_path, features=features, labels=labels)
    print(f"Dataset saved to: {file_path}")


def _get_filenames(labels_df: pd.DataFrame, augmented: bool) -> list:
    """
    Retrieves filenames from the labels dataframe based on augmentation flag.

    Args:
        labels_df (pd.DataFrame): Labels dataframe.
        augmented (bool): Whether to retrieve augmented filenames.

    Returns:
        list: List of filenames.
    """
    if augmented:
        augmented_labels_df = pd.read_csv(CONFIG["augmented_label_path"])
        return augmented_labels_df["filename"].tolist()
    else:
        return labels_df["filename"].tolist()


def _get_story_type(filename: str, labels_df: pd.DataFrame, augmented: bool) -> str:
    """
    Retrieves the story type for a given filename.

    Args:
        filename (str): Filename.
        labels_df (pd.DataFrame): Labels dataframe.
        augmented (bool): Whether to use augmented labels.

    Returns:
        str: Story type.
    """
    if augmented:
        augmented_labels_df = pd.read_csv(CONFIG["augmented_label_path"])
        return augmented_labels_df[augmented_labels_df["filename"] == filename]["Story_type"].values[0]
    else:
        return labels_df[labels_df["filename"] == filename]["Story_type"].values[0]


def _get_language(filename: str, labels_df: pd.DataFrame, augmented: bool) -> str:
    """
    Retrieves the language for a given filename.

    Args:
        filename (str): Filename.
        labels_df (pd.DataFrame): Labels dataframe.
        augmented (bool): Whether to use augmented labels.

    Returns:
        str: Language.
    """
    if augmented:
        augmented_labels_df = pd.read_csv(CONFIG["augmented_label_path"])
        return augmented_labels_df[augmented_labels_df["filename"] == filename]["Language"].values[0]
    else:
        return labels_df[labels_df["filename"] == filename]["Language"].values[0]


def load_data(dataset_path: str, labels_df: pd.DataFrame, augmented: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and extracts features and labels from the dataset.

    Args:
        dataset_path (str): Path to the dataset.
        labels_df (pd.DataFrame): Labels dataframe.
        augmented (bool, optional): Whether to load augmented data. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of features and labels.
    """
    filenames = _get_filenames(labels_df, augmented)
    features, labels = [], []

    for filename in tqdm(filenames, desc="Loading data", ncols=100, unit="file"):
        file_path = os.path.join(dataset_path, filename)
        audio_features = DataTransformer.extract_features(file_path)

        story_type = _get_story_type(filename, labels_df, augmented)
        language = _get_language(filename, labels_df, augmented)

        language_feature = np.ones((1, audio_features.shape[1])) if language == "English" else np.zeros((1, audio_features.shape[1]))

        audio_features = np.vstack([audio_features, language_feature])

        features.append(audio_features)
        labels.append(story_type)

    return np.array(features), np.array(labels)
