import os
from typing import Tuple
from .base import BaseModel
from keras import layers, models, callbacks
from utils import show_history


class NNModel(BaseModel):
    def __init__(self):
        super().__init__()

    def build_model(self, input_shape: Tuple[int, int, int]):
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

    def train(self, X_train, y_train, x_val, y_val, num_of_models, epochs=10, batch_size=32):
        print(f"Training Model {num_of_models}")
        
        early_stopping = callbacks.EarlyStopping(monitor="loss", patience=100)
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(self.config.model_path, f"best_model_{num_of_models}.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
        callbacks_list = [early_stopping, model_checkpoint]

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

    def predict(self, X_test):
        return self.model.predict(X_test).flatten()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = models.load_model(path)
        return self.model
