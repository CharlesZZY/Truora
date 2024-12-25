import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import Config
from .data_transformer import DataTransformer
from .evaluator import Evaluator
from .predictor import Predictor


class Pipeline:
    def __init__(self, config: Config, data_transformer: DataTransformer = None, evaluator: Evaluator = None, predictor: Predictor = None):
        self.config = config
        self.label_encoder: LabelEncoder = None
        self.data_transformer = data_transformer
        self.evaluator = evaluator
        self.predictor = predictor

    def init_label_encoder(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.config.labels)

    def pre_process(self, load_from_npz: bool = False, augmented: bool = False):
        if load_from_npz:
            original_data = self.data_transformer.load_data_from_npz(os.path.join(self.config.dataset_path, "original_data.npz"))
            augmented_data = self.data_transformer.load_data_from_npz(os.path.join(self.config.dataset_path, "augmented_data.npz")) if augmented else None
        else:
            labels_df = pd.read_csv(self.config.label_path)
            original_data = self.data_transformer.load_data(self.config.story_path, labels_df, augmented=False)
            augmented_data = self.data_transformer.load_data(self.config.augmented_story_path, labels_df, augmented=True) if augmented else None

        datasets = []

        for i in range(self.config.n_models):
            if augmented and augmented_data is not None:
                sampled_idx_original = np.random.choice(original_data['features'].shape[0], original_data['features'].shape[0], replace=True)
                sampled_idx_augmented = np.random.choice(augmented_data['features'].shape[0], augmented_data['features'].shape[0], replace=True)

                sampled_original_features = original_data['features'][sampled_idx_original]
                sampled_original_labels = original_data['labels'][sampled_idx_original]

                sampled_augmented_features = augmented_data['features'][sampled_idx_augmented]
                sampled_augmented_labels = augmented_data['labels'][sampled_idx_augmented]

                features = np.concatenate((sampled_original_features, sampled_augmented_features), axis=0)
                labels = np.concatenate((sampled_original_labels, sampled_augmented_labels), axis=0)
            else:
                sampled_idx_original = np.random.choice(original_data['features'].shape[0], original_data['features'].shape[0], replace=True)

                features = original_data['features'][sampled_idx_original]
                labels = original_data['labels'][sampled_idx_original]

            X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]
            X_test = X_test[..., np.newaxis]

            datasets.append((X_train, y_train, X_val, y_val, X_test, y_test))

        self.config.labels = np.unique(labels)

        print(f"Labels: {self.config.labels}")
        print(f"Original dataset size: {original_data['features'].shape[0]}")
        if augmented:
            print(f"Augmented dataset size: {augmented_data['features'].shape[0]}")
        print(f"Number of datasets created: {len(datasets)}")

        for i, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(datasets):
            y_train_encoded = self.label_encoder.transform(y_train)
            y_val_encoded = self.label_encoder.transform(y_val)
            y_test_encoded = self.label_encoder.transform(y_test)

            datasets[i] = (X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded)

        return datasets

    def nn_train(self, datasets, continue_training: bool = False):
        self.nn_models = []
        for i in range(self.config.num_nn_models):
            X_train, y_train, X_val, y_val, _, _ = datasets[i]
            model = self.trainer.train_nn(num_of_models=i+1, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, continue_training=continue_training)
            self.nn_models.append(model)

    def train_model(self, model, datasets):
        if isinstance(model, list):
            for i in range(len(model)):
                X_train, y_train, X_val, y_val, _, _ = datasets[i]
                model[i].train(X_train, y_train)
        else:
            for i in range(len(datasets)):
                X_train, y_train, X_val, y_val, _, _ = datasets[i]
                model.train(X_train, y_train)

    def evaluate(self, X_test, y_test):
        labels = self.config.labels
        models_to_use = []

        if self.use_nn and self.nn_models is not None:
            models_to_use.append(self.nn_models)
        if self.use_rf and self.rf_model is not None:
            models_to_use.append(self.rf_model)
        if self.use_knn and self.knn_model is not None:
            models_to_use.append(self.knn_model)
        if self.use_lr and self.lr_model is not None:
            models_to_use.append(self.lr_model)
        if self.use_svm and self.svm_model is not None:
            models_to_use.append(self.svm_model)

        if not models_to_use:
            raise ValueError("At least one model (NN or RF) must be enabled for evaluation.")

        final_pred, final_pred_prob = self.evaluator.evaluate_ensemble_model(
            X_test=X_test,
            y_true=y_test,
            labels=labels,
            model_list=self.nn_models if self.use_nn else None,
            rf_model=self.rf_model if self.use_rf else None,
            knn_model=self.knn_model if self.use_knn else None,
            lr_model=self.lr_model if self.use_lr else None,
            svm_model=self.svm_model if self.use_svm else None
        )

        predicted_labels = self.label_encoder.inverse_transform(final_pred)
        true_labels = self.label_encoder.inverse_transform(y_test)

        df = pd.DataFrame({
            'Actual': true_labels,
            'Predicted': predicted_labels
        })
        print(df)
