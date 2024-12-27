import os
os.environ["KERAS_BACKEND"] = "torch"
from .base import BaseModel
from utils.plotter import show_history
from typing import Any, Tuple
from keras import layers, models, callbacks


class NNModel(BaseModel):
    """
    Neural Network model using CNN-LSTM architecture with Attention mechanism.
    """

    def __init__(self) -> None:
        """
        Initializes the NNModel.
        """
        super().__init__()

    def build(self, input_shape: Tuple[int, int, int]) -> models.Model:
        """
        Builds the CNN-LSTM model architecture.

        Args:
            input_shape (Tuple[int, int, int]): Shape of the input data.

        Returns:
            models.Model: Compiled Keras model.
        """
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
        self.model = model
        return self.model

    def train(self, X_train: Any, y_train: Any, x_val: Any, y_val: Any, model_path: str, epochs: int = 10, batch_size: int = 32) -> models.Model:
        """
        Trains the Neural Network model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            x_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            model_path (str): Path to save the best model.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 32.

        Returns:
            models.Model: Trained Keras model.
        """
        print(f"Training Neural Network ...")

        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        early_stopping = callbacks.EarlyStopping(monitor="val_accuracy", patience=100)
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, f"nn_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
        callbacks_list = [lr_scheduler, early_stopping, model_checkpoint]

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list
        )
        show_history(history)

        return self.model

    def predict(self, X_test: Any) -> Any:
        """
        Generates predictions for the test data.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Flattened prediction probabilities.
        """
        return self.model.predict(X_test).flatten()

    def save(self, path: str) -> None:
        """
        Saves the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)

    def load(self, path: str) -> models.Model:
        """
        Loads the model from the specified path.

        Args:
            path (str): Path to load the model from.

        Returns:
            models.Model: Loaded Keras model.
        """
        self.model = models.load_model(path)
        return self.model
