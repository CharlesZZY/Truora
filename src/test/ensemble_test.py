import pytest
from sklearn.calibration import LabelEncoder
from config import Config
from utils.data_transformer import DataTransformer
import numpy as np

config = Config(
    train_dataset="datasets/train.npz",
    validation_dataset="datasets/validation.npz",
    test_dataset="datasets/test.npz",
    labels=["True Story", "Deceptive Story"],
    model_path="models/",
    epochs=2,
    batch_size=10
)


@pytest.fixture
def label_encoder():
    label_encoder = LabelEncoder()
    label_encoder.fit(config.labels)
    return label_encoder


@pytest.fixture
def data(label_encoder):
    data_transformer = DataTransformer(config)
    X_train, y_train, X_val, y_val, X_test, y_test = data_transformer.get_datasets()
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train[:10], y_train_encoded[:10], X_val[:10], y_val_encoded[:10], X_test[:10], y_test_encoded[:10]


@pytest.fixture
def models():
    from model import NNModel, SVM, RFModel, LRModel, KNNModel
    nn_model = NNModel()
    rf_model = RFModel()
    knn_model = KNNModel()
    lr_model = LRModel()
    svm_model = SVM()
    nn_model.load("models/nn_model.keras")
    rf_model.load("models/rf_model.joblib")
    knn_model.load("models/knn_model.joblib")
    lr_model.load("models/lr_model.joblib")
    svm_model.load("models/svm_model.joblib")
    return {
        "nn_model": nn_model,
        "rf_model": rf_model,
        "knn_model": knn_model,
        "lr_model": lr_model,
        "svm_model": svm_model,
    }


def test_ensemble(data, models):
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = data
    from model import EnsembleModel
    ensemble = EnsembleModel(models)
    ensemble.build()
    ensemble.train(train_features, train_labels)
    predictions = ensemble.predict(test_features)
    assert len(predictions) == len(test_labels)
