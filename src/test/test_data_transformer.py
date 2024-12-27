from utils.data_transformer import DataTransformer
from config import Config

config = Config(
    train_dataset="datasets/train.npz",
    validation_dataset="datasets/validation.npz",
    test_dataset="datasets/test.npz",
    labels=["True Story", "Deceptive Story"],
    model_path="models/",
    epochs=2,
    batch_size=10
)


def test_get_datasets():
    data_transformer = DataTransformer(config)
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = data_transformer.get_datasets()
    assert train_features is not None
    assert train_labels is not None
    assert validation_features is not None
    assert validation_labels is not None
    assert test_features is not None
    assert test_labels is not None
