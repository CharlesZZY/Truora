import os
os.environ["KERAS_BACKEND"] = "torch"

from utils.augment import load_data, process_audio, save_data
from utils.data_transformer import split_and_save_data
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
from config import Config
from utils.data_transformer import DataTransformer
from model import NNModel, RFModel, KNNModel, LRModel, SVM
from utils.evaluator import Evaluator
from utils.predictor import Predictor
from pipeline import Pipeline
from utils.plotter import plot_decision_boundary, plot_confusion_matrix, plot_roc_curve
from model.ensemble import EnsembleModel
from utils.model_builder import ModelBuilder
from typing import *


CONFIG = {
    "dataset_path": "../datasets/",
    "story_path": "../datasets/CBU5201DD_stories/",
    "label_path": "../datasets/CBU5201DD_stories_attributes.csv",
    "augmented_story_path": "../datasets/CBU5201DD_stories/_augmented/",
    "augmented_label_path": "../datasets/CBU5201DD_stories_attributes_augmented.csv",
}

label_df = pd.read_csv(CONFIG['label_path'])
filenames = label_df['filename'].tolist()
languages = label_df['Language'].tolist()

augmented_filenames = []
augmented_labels = []
augmented_languages = []

augmented_story_path = CONFIG['story_path'] + "_augmented/"
if not os.path.exists(augmented_story_path):
    os.makedirs(augmented_story_path)

for filename, language in tqdm(zip(filenames, languages), desc="Processing audio files", unit="file", total=len(filenames)):
    augmented_audios = process_audio(filename, CONFIG['story_path'])

    for i, y in enumerate(augmented_audios):
        augmented_filename = filename.split('.')[0] + f"_augmented_{i}.wav"
        augmented_filenames.append(augmented_filename)
        augmented_languages.append(language)
        augmented_labels.append(label_df[label_df['filename'] == filename]['Story_type'].values[0])

        sf.write(os.path.join(augmented_story_path, augmented_filename), y, 16000)

augmented_labels_df = pd.DataFrame({
    "filename": augmented_filenames,
    "Language": augmented_languages,
    "Story_type": augmented_labels
})

augmented_labels_df.to_csv(CONFIG['label_path'].replace('.csv', '_augmented.csv'), index=False)

print(f"Data augmentation complete. The new label file is saved at: {CONFIG['label_path'].replace('.csv', '_augmented.csv')}")

labels_df = pd.read_csv(CONFIG['label_path'])
original_features, original_labels = load_data(CONFIG['story_path'], labels_df, augmented=False)
augmented_features, augmented_labels = load_data(CONFIG['augmented_story_path'], labels_df, augmented=True)

save_data(original_features, original_labels, os.path.join(CONFIG["dataset_path"], 'original_data.npz'))
save_data(augmented_features, augmented_labels, os.path.join(CONFIG["dataset_path"], 'augmented_data.npz'))


split_and_save_data(CONFIG['dataset_path'] + 'original_data.npz', CONFIG['dataset_path'] + 'augmented_data.npz', CONFIG['dataset_path'], train_ratio=0.7, val_ratio=0.15)

# Define the paths to the split datasets
train_path = CONFIG['dataset_path'] + 'train.npz'
val_path = CONFIG['dataset_path'] + 'validation.npz'
test_path = CONFIG['dataset_path'] + 'test.npz'

# Load the split datasets
train_data = np.load(train_path)
val_data = np.load(val_path)
test_data = np.load(test_path)

train_features = train_data['features']
train_labels = train_data['labels']

val_features = val_data['features']
val_labels = val_data['labels']

test_features = test_data['features']
test_labels = test_data['labels']

# Create DataFrames for easier manipulation
train_df = pd.DataFrame(train_labels, columns=['Label'])
val_df = pd.DataFrame(val_labels, columns=['Label'])
test_df = pd.DataFrame(test_labels, columns=['Label'])

train_data["features"].shape, val_data["features"].shape, test_data["features"].shape, train_data["labels"].shape, val_data["labels"].shape, test_data["labels"].shape

label_counts = {
    "Dataset": ["Train", "Validation", "Test"],
    "True Story": [
        (train_df["Label"] == "True Story").sum(),
        (val_df["Label"] == "True Story").sum(),
        (test_df["Label"] == "True Story").sum(),
    ],
    "Deceptive Story": [
        (train_df["Label"] == "Deceptive Story").sum(),
        (val_df["Label"] == "Deceptive Story").sum(),
        (test_df["Label"] == "Deceptive Story").sum(),
    ],
}
label_df = pd.DataFrame(label_counts).set_index("Dataset")

plt.figure(figsize=(8, 6))
sns.heatmap(label_df, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
plt.title("Label Distribution Across Datasets")
plt.xlabel("Label")
plt.ylabel("Dataset")
plt.show()


sampled_features = train_features.mean(axis=2)
features_df = pd.DataFrame(sampled_features, columns=[f"Feature {i}" for i in range(train_features.shape[1])])

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
sampled_columns_front = features_df.columns[:10]
sns.boxplot(data=features_df[sampled_columns_front])
plt.title("Feature Distribution (First 10 Features)")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(axis="y")

plt.subplot(1, 2, 2)
sampled_columns_back = features_df.columns[-10:]
sns.boxplot(data=features_df[sampled_columns_back])
plt.title("Feature Distribution (Last 10 Features)")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(axis="y")

plt.tight_layout()
plt.show()

flattened_features = train_features.mean(axis=2)
labels = train_labels

pca = PCA(n_components=2)
pca_result = pca.fit_transform(flattened_features)

pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Label"] = labels

plt.figure(figsize=(12, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Label", palette="Set2", s=60)
plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Label")
plt.grid()
plt.show()

flattened_features = train_features.mean(axis=2)
labels = train_labels

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(flattened_features)

tsne_df = pd.DataFrame(tsne_result, columns=["Dim1", "Dim2"])
tsne_df["Label"] = labels

plt.figure(figsize=(12, 6))
sns.scatterplot(data=tsne_df, x="Dim1", y="Dim2", hue="Label", palette="Set2", s=60)
plt.title("t-SNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title="Label")
plt.grid()
plt.show()

sampled_features = train_features.mean(axis=2)
features_df = pd.DataFrame(sampled_features, columns=[f"Feature {i}" for i in range(train_features.shape[1])])

correlation_matrix_front = features_df.iloc[:, :30].corr()

correlation_matrix_back = features_df.iloc[:, -30:].corr()

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix_front, cmap="coolwarm", annot=False, fmt=".2f", square=True, cbar=True)
plt.title("Feature Correlation Matrix (First 30 Features)")
plt.xlabel("Feature Index")
plt.ylabel("Feature Index")

plt.subplot(1, 2, 2)
sns.heatmap(correlation_matrix_back, cmap="coolwarm", annot=False, fmt=".2f", square=True, cbar=True)
plt.title("Feature Correlation Matrix (Last 30 Features)")
plt.xlabel("Feature Index")
plt.ylabel("Feature Index")

plt.tight_layout()
plt.show()

config = Config(
    train_dataset="../datasets/train.npz",
    validation_dataset="../datasets/validation.npz",
    test_dataset="../datasets/test.npz",
    model_path="../models/",
    epochs=200,
    batch_size=10,
    labels=["True Story", "Deceptive Story"]
)

data_transformer = DataTransformer(config)
model_builder = ModelBuilder()
evaluator = Evaluator()
predictor = Predictor()
model_list = {
    "nn_model": NNModel(),
    "rf_model": RFModel(),
    "knn_model": KNNModel(),
    "lr_model": LRModel(),
    "svm_model": SVM(),
}

pipeline = Pipeline(config=config, data_transformer=data_transformer, evaluator=evaluator, predictor=predictor, model_list=model_list)

datasets = pipeline.data_preprocessing()

X_train, y_train, X_val, y_val, X_test, y_test = datasets

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

nn_model = model_builder.build_nn_model(input_shape=X_train[0].shape)
pipeline.train_model(model_name="nn_model", model=nn_model, X_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, model_path=config.model_path, epochs=config.epochs, batch_size=config.batch_size)
nn_model.model.summary()

# Flatten the data
X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)  # Shape: (420, 167, 7500)

# Aggregate across the time axis
X_train_agg = X_train_flat.mean(axis=2)  # Shape: (420, 167)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_agg)  # Shape: (420, 2)


rf_model = model_builder.build_rf_model(random_state=42, n_estimators=100)
pipeline.train_model(model_name="rf_model", model=rf_model, X_train=X_train, y_train=y_train)
rf_model.save(os.path.join(config.model_path, "rf_model.joblib"))
rf_model.model

plot_decision_boundary(rf_model.model, X_train_pca, y_train, "Random Forest Decision Boundary")

knn_model = model_builder.build_knn_model(n_neighbors=3)
pipeline.train_model(model_name="knn_model", model=knn_model, X_train=X_train, y_train=y_train)
knn_model.save(os.path.join(config.model_path, "knn_model.joblib"))
knn_model.model

plot_decision_boundary(knn_model.model, X_train_pca, y_train, "KNN Decision Boundary")

lr_model = model_builder.build_logistic_regression_model(C=1, random_state=42, max_iter=100)
pipeline.train_model(model_name="lr_model", model=lr_model, X_train=X_train, y_train=y_train)
lr_model.save(os.path.join(config.model_path, "lr_model.joblib"))
lr_model.model

plot_decision_boundary(lr_model.model, X_train_pca, y_train, "Logistic Regression Decision Boundary")

svm_model = model_builder.build_svm_model(C=1.0, kernel="rbf")
pipeline.train_model(model_name="svm_model", model=svm_model, X_train=X_train, y_train=y_train)
svm_model.save(os.path.join(config.model_path, "svm_model.joblib"))
svm_model.model

plot_decision_boundary(svm_model.model, X_train_pca, y_train, "SVM Decision Boundary")

pipeline.load_model("nn_model", os.path.join(config.model_path, f"nn_model.keras"))

pipeline.load_model("rf_model", os.path.join(config.model_path, "rf_model.joblib"))

pipeline.load_model("knn_model", os.path.join(config.model_path, "knn_model.joblib"))

pipeline.load_model("lr_model", os.path.join(config.model_path, "lr_model.joblib"))

pipeline.load_model("svm_model", os.path.join(config.model_path, "svm_model.joblib"))

pipeline.evaluate(X_test, y_test, test_models=["nn_model"])

pipeline.evaluate(X_test, y_test, test_models=["rf_model"])

pipeline.evaluate(X_test, y_test, test_models=["knn_model"])

pipeline.evaluate(X_test, y_test, test_models=["lr_model"])

pipeline.evaluate(X_test, y_test, test_models=["svm_model"])

pipeline.evaluate(X_test, y_test, test_models=["nn_model", "rf_model", "knn_model", "lr_model", "svm_model"])

pipeline.predict_audio("../datasets/CBU5201DD_stories/00001.wav", language="Chinese", use_models=["nn_model", "rf_model", "knn_model", "lr_model", "svm_model"])
pipeline.predict_audio("../datasets/CBU5201DD_stories/00017.wav", language="English", use_models=["nn_model", "rf_model", "knn_model", "lr_model", "svm_model"])

models = {
    "nn_model": nn_model,
    "rf_model": rf_model,
    "knn_model": knn_model,
    "lr_model": lr_model,
    "svm_model": svm_model,
}

ensemble_model = EnsembleModel(models=models)

ensemble_model.weight_model

ensemble_model.train(X_train=X_train, y_train=y_train)

ensemble_pred_prob = ensemble_model.predict(X_test)
ensemble_pred = (ensemble_pred_prob > 0.5).astype(int)

print("Ensemble Model Performance:")
print(classification_report(y_test, ensemble_pred))
confusion_matrix_display = confusion_matrix(y_test, ensemble_pred)
Evaluator.evaluate_model(y_test, ensemble_pred)
plot_confusion_matrix(y_test, ensemble_pred, labels=config.labels)
plot_roc_curve(y_test, ensemble_pred_prob)

# Check if the ensemble model has a weight model and if it has coefficients
if hasattr(ensemble_model, "weight_model") and hasattr(ensemble_model.weight_model, "coef_"):
    model_weights = ensemble_model.weight_model.coef_[0]
    model_names = list(models.keys())

    # Normalize the weights
    total_weight = sum(model_weights)
    normalized_weights = model_weights / total_weight

    # Print the trained model weights
    print("Trained Model Weights:")
    for model_name, weight in zip(model_names, normalized_weights):
        print(f"{model_name}: {weight:.4f}")
else:
    print("Ensemble model has not been trained or weight_model is not properly initialized.")
