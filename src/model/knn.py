from .base import BaseModel
from sklearn.neighbors import KNeighborsClassifier
import joblib


class KNNModel(BaseModel):
    def __init__(self):
        super().__init__()

    def build_model(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return self.model

    def train(self, X_train, y_train):
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_2d, y_train)
        return self.model

    def predict(self, X_test):
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test_2d)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self.model
