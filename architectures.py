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

    def __get_callbacks(self,path_model) :
        callbacks_baseline = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(path_model, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        return callbacks_baseline

    
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
            callbacks=self.__get_callbacks("best_model_simple.h5"),
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
            callbacks=self.__get_callbacks("best_model_profond.h5"),
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
                callbacks=self.__get_callbacks("best_model_transfer_learning.h5"),
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
            callbacks=self.__get_callbacks("best_model_tf_ft_EfficientNetB0.h5"),
            verbose=1
        )

        return history_tf_ft_EfficientNetB0, model_tf_ft_EfficientNetB0


if __name__ == "__main__":

    from tensorflow.keras.utils import to_categorical

    print("Chargement de CIFAR-100...")

    # 1) Chargement des données
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # 2) Normalisation
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 3) One-hot encoding
    y_train_ohe = to_categorical(y_train, 100)
    y_test_ohe = to_categorical(y_test, 100)

    print("Shapes :")
    print("X_train :", X_train.shape)
    print("y_train_ohe :", y_train_ohe.shape)
    print("X_test :", X_test.shape)
    print("y_test_ohe :", y_test_ohe.shape)

    # 4) Instanciation de la classe
    archi = Architectures(X_train, y_train_ohe)

    # --------------------------------------------------
    # Choisis ici le modèle à tester
    # --------------------------------------------------
    modele_a_tester = "simple"
    # valeurs possibles :
    # "simple"
    # "profond"
    # "mobilenet"
    # "efficientnet"

    if modele_a_tester == "simple":
        print("\n===== TEST CNN SIMPLE =====")
        history, model = archi.cnn_simple()

    elif modele_a_tester == "profond":
        print("\n===== TEST CNN PROFOND =====")
        history, model = archi.cnn_profond()

    elif modele_a_tester == "mobilenet":
        print("\n===== TEST TRANSFER LEARNING MobileNetV2 =====")
        history, model = archi.cnn_transferlearning_MobileNetV2()

    elif modele_a_tester == "efficientnet":
        print("\n===== TEST FINE-TUNING EfficientNetB0 =====")
        history, model = archi.cnn_transferlearning_finetuning_EfficientNetB0()

    else:
        raise ValueError("Valeur invalide pour 'modele_a_tester'")

    # 5) Évaluation sur le jeu de test
    print("\nÉvaluation sur le jeu de test...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_ohe, verbose=1)

    print(f"\nTest loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")

    # 6) Affichage des métriques d'entraînement
    print("\nClés présentes dans history.history :")
    print(history.history.keys())

    # 7) Courbes accuracy / loss
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    