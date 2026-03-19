import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2,EfficientNetB0
import tensorflow as tf
from tensorflow.keras.datasets import cifar100   # ← si tu fais CIFAR-100
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
    GlobalAveragePooling2D
)
from tensorflow.keras import optimizers, callbacks
    # Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class Architectures :


    def __init__(self, X_train_norm,y_train_onehot) :
        self.X_train_normalized = X_train_norm
        self.y_train_ohe = y_train_onehot
        self.callbacks_baseline = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint('baseline_cnn_cifar100.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]


    
    def cnn_simple(self) :
        # Modèle CNN simple
        model_simple = Sequential([
            Conv2D(64, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.25),

            Conv2D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.3),

            Conv2D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.4),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax')  # 100 classes!
        ])

        model_simple.compile(
            optimizer=optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model_simple.summary()

        # Entraîner le modèle
        history_simple = model_simple.fit(
            self.X_train_normalized, self.y_train_ohe,
            batch_size=32,
            epochs=15,
            validation_split=0.2,
            callbacks=self.callbacks_baseline,
            verbose=1
        )

        return history_simple, model_simple
    

    def cnn_profond(self) :
        # Modèle CNN simple
        model_profond = Sequential([

            Conv2D(64, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
            BatchNormalization(),
            Conv2D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.25),

            Conv2D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.3),

            Conv2D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.4),

            Conv2D(512, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.4),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax')
        ])

        model_profond.compile(
            optimizer=optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model_profond.summary()

        # Entraîner le modèle
        history_profond = model_profond.fit(
            self.X_train_normalized, self.y_train_ohe,
            batch_size=32,
            epochs=15,
            validation_split=0.2,
            callbacks=self.callbacks_baseline,
            verbose=1
        )

        return history_profond, model_profond

    def cnn_transferlearning_MobileNetV2(self) :
            
            base_model = MobileNetV2(
                input_shape=(96, 96, 3),  # ou (224, 224, 3)
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False

            model_transfer_learning = Sequential([
                layers.Resizing(96, 96),  # Redimensionner à l'entrée
                base_model,
                GlobalAveragePooling2D(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(100, activation='softmax')  # 100 classes!
            ])

            model_transfer_learning.compile(
                optimizer=optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Entraîner le modèle
            history_transfer_learning = model_transfer_learning.fit(
                self.X_train_normalized, self.y_train_ohe,
                batch_size=32,
                epochs=15,
                validation_split=0.2,
                callbacks=self.callbacks_baseline,
                verbose=1
            )

            return history_transfer_learning, model_transfer_learning

    def cnn_transferlearning_finetuning_EfficientNetB0(self) :
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),  # Redimensionner CIFAR-100
            include_top=False,
            weights='imagenet'
        )

        # Fine-tuning : dégeler couches finales
        base_model.trainable = True
        for layer in base_model.layers[:-50]:  # Dégeler plus de couches
            layer.trainable = False

        model_tf_ft_EfficientNetB0 = Sequential([
            layers.Resizing(224, 224),  # Redimensionner à l'entrée
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax')  # 100 classes!
        ])

        model_tf_ft_EfficientNetB0.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

        # Entraîner le modèle
        history_tf_ft_EfficientNetB0 = model_tf_ft_EfficientNetB0.fit(
            self.X_train_normalized, self.y_train_ohe,
            batch_size=32,
            epochs=15,
            validation_split=0.2,
            callbacks=self.callbacks_baseline,
            verbose=1
        )

        return history_tf_ft_EfficientNetB0, model_tf_ft_EfficientNetB0


if __name__ == "__main__":

    print("Chargement des données CIFAR-100...")

    # 1. Chargement des données
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # 2. Normalisation
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 3. One-hot encoding
    from tensorflow.keras.utils import to_categorical
    y_train_ohe = to_categorical(y_train, 100)
    y_test_ohe = to_categorical(y_test, 100)

    print("Données prêtes :", X_train.shape, y_train_ohe.shape)

    # 4. Initialisation de ta classe
    archi = Architectures(X_train, y_train_ohe)

    # ===============================
    # 5. TESTS DES MODÈLES
    # ===============================

    # 🔹 CNN SIMPLE
    print("\n===== CNN SIMPLE =====")
    history_simple, model_simple = archi.cnn_simple()

    # Évaluation
    loss, acc = model_simple.evaluate(X_test, y_test_ohe, verbose=0)
    print(f"CNN Simple - Test accuracy: {acc:.4f}")

    # 🔹 CNN PROFOND
    print("\n===== CNN PROFOND =====")
    history_profond, model_profond = archi.cnn_profond()

    loss, acc = model_profond.evaluate(X_test, y_test_ohe, verbose=0)
    print(f"CNN Profond - Test accuracy: {acc:.4f}")

    # 🔹 MOBILENETV2
    print("\n===== TRANSFER LEARNING MobileNetV2 =====")
    history_mobilenet, model_mobilenet = archi.cnn_transferlearning_MobileNetV2()

    loss, acc = model_mobilenet.evaluate(X_test, y_test_ohe, verbose=0)
    print(f"MobileNetV2 - Test accuracy: {acc:.4f}")

    # 🔹 EFFICIENTNETB0
    print("\n===== FINE-TUNING EfficientNetB0 =====")
    history_effnet, model_effnet = archi.cnn_transferlearning_finetuning_EfficientNetB0()

    loss, acc = model_effnet.evaluate(X_test, y_test_ohe, verbose=0)
    print(f"EfficientNetB0 - Test accuracy: {acc:.4f}")

    print("\n🎯 Tests terminés !")



