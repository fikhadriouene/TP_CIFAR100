import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import optimizers

# Import Phase 3
from architectures import Architectures        

# ──────────────────────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR     = "phase4_outputs"
EPOCHS_FULL    = 100        # epochs max (EarlyStopping stoppe avant si nécessaire)
BATCH_SIZE     = 64
LEARNING_RATE  = 0.001
NUM_CLASSES    = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_KEYS = {
    "CNN Simple"       : "simple",
    "CNN Profond"      : "profond",
    "MobileNetV2"      : "mobilenet",
    "EfficientNetB0"   : "efficientnet",
}


# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT & AUGMENTATION DES DONNÉES
# ══════════════════════════════════════════════════════════════

def load_data():
    """Charge et prépare CIFAR-100 (normalisation + one-hot). Partie de Damien ?"""
    print("Chargement CIFAR-100...")
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    y_train_ohe = to_categorical(y_train, NUM_CLASSES)
    y_test_ohe  = to_categorical(y_test,  NUM_CLASSES)

    print(f"   X_train={X_train.shape}  X_test={X_test.shape}")
    return X_train, y_train_ohe, X_test, y_test_ohe


def make_augmented_data(X_train, y_train_ohe, level="medium"):
    """
    Retourne (X_aug, y_aug) après data augmentation en mémoire.
    level : 'light' | 'medium' | 'aggressive'
    """
    configs = {
        "light": dict(horizontal_flip=True,
                      width_shift_range=0.1, height_shift_range=0.1),
        "medium": dict(horizontal_flip=True,
                       width_shift_range=0.15, height_shift_range=0.15,
                       rotation_range=15, zoom_range=0.1),
        "aggressive": dict(horizontal_flip=True,
                           width_shift_range=0.2, height_shift_range=0.2,
                           rotation_range=20, zoom_range=0.2,
                           shear_range=0.1, channel_shift_range=20.0),
    }
    gen = ImageDataGenerator(**configs.get(level, {}))
    gen.fit(X_train)

    # Génère autant d'exemples augmentés que le dataset original
    batches, count = [], 0
    for Xb, yb in gen.flow(X_train, y_train_ohe, batch_size=512, shuffle=False):
        batches.append((Xb, yb))
        count += len(Xb)
        if count >= len(X_train):
            break

    X_aug = np.concatenate([b[0] for b in batches])[:len(X_train)]
    y_aug = np.concatenate([b[1] for b in batches])[:len(y_train_ohe)]

    # Mélange original + augmenté
    X_combined = np.concatenate([X_train, X_aug])
    y_combined = np.concatenate([y_train_ohe, y_aug])
    idx = np.random.permutation(len(X_combined))
    print(f"   Data augmentation '{level}' → {len(X_combined)} exemples")
    return X_combined[idx], y_combined[idx]


# ══════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT PHASE 4
#    Réutilisation classe Phase 3 mais recompilation des modèles
#    avec les hyperparamètres Phase 4 (plus d'epochs, LR explicite…)
# ══════════════════════════════════════════════════════════════

def _get_callbacks_p4(model_name: str):
    """Callbacks Phase 4 (patience augmentée, checkpoint .keras)."""
    d = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(d, "best_model.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
    ]


def _recompile(model):
    """Recompile avec LR Phase 4 + métrique Top-5."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )
    return model


def train_all_models(X_train, y_train_ohe, augmentation_level="medium"):
    """
    Construit les 4 modèles via la classe Phase 3,
    puis les entraîne avec les paramètres Phase 4.
    Retourne { nom_lisible: (history, model) }
    """
    # Augmentation
    X_aug, y_aug = make_augmented_data(X_train, y_train_ohe, augmentation_level)

    # Instanciation Phase 3 (avec données augmentées)
    archi = Architectures(X_aug, y_aug)

    results = {}

    # ── CNN Simple ──────────────────────────────────────────
    print("\n CNN Simple")
    _, model = archi.cnn_simple()           # entraîne une première fois (15 ep. Phase 3)
    model = _recompile(model)
    h = model.fit(X_aug, y_aug,
                  batch_size=BATCH_SIZE, epochs=EPOCHS_FULL,
                  validation_split=0.2,
                  callbacks=_get_callbacks_p4("cnn_simple"), verbose=1)
    results["CNN Simple"] = (h, model)
    _save_history("cnn_simple", h)

    # ── CNN Profond ──────────────────────────────────────────
    print("\n CNN Profond")
    _, model = archi.cnn_profond()
    model = _recompile(model)
    h = model.fit(X_aug, y_aug,
                  batch_size=BATCH_SIZE, epochs=EPOCHS_FULL,
                  validation_split=0.2,
                  callbacks=_get_callbacks_p4("cnn_profond"), verbose=1)
    results["CNN Profond"] = (h, model)
    _save_history("cnn_profond", h)

    # ── MobileNetV2 ──────────────────────────────────────────
    print("\n  MobileNetV2")
    _, model = archi.cnn_transferlearning_MobileNetV2()
    model = _recompile(model)
    h = model.fit(X_aug, y_aug,
                  batch_size=BATCH_SIZE, epochs=EPOCHS_FULL,
                  validation_split=0.2,
                  callbacks=_get_callbacks_p4("mobilenetv2"), verbose=1)
    results["MobileNetV2"] = (h, model)
    _save_history("mobilenetv2", h)

    # ── EfficientNetB0 ───────────────────────────────────────
    print("\n  EfficientNetB0 (fine-tuning)")
    _, model = archi.cnn_transferlearning_finetuning_EfficientNetB0()
    model = _recompile(model)
    h = model.fit(X_aug, y_aug,
                  batch_size=BATCH_SIZE, epochs=EPOCHS_FULL,
                  validation_split=0.2,
                  callbacks=_get_callbacks_p4("efficientnetb0"), verbose=1)
    results["EfficientNetB0"] = (h, model)
    _save_history("efficientnetb0", h)

    return results


def _save_history(name: str, history):
    """Exporte l'historique en JSON."""
    path = os.path.join(OUTPUT_DIR, name, "history.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convertit float32 → float natif pour JSON
    serializable = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Historique → {path}")


# ══════════════════════════════════════════════════════════════
# 3. TUNING ADAPTATIF
# ══════════════════════════════════════════════════════════════

def adaptive_tuning_report(name: str, history) -> dict:
    """
    Analyse les courbes et retourne un dict de recommandations.
    """
    h           = history.history
    train_acc   = h["accuracy"]
    val_acc     = h["val_accuracy"]
    val_loss    = h["val_loss"]
    n           = len(val_loss)

    gap         = train_acc[-1] - val_acc[-1]           # overfitting gap
    stagnation  = (val_loss[-6] - val_loss[-1]) if n >= 6 else 1.0  # amélioration récente

    recs = {}

    # — Learning rate —
    if stagnation < 0.002:
        recs["learning_rate"] = (
            "Val_loss stagne → LR trop élevé ou plateau. "
            "ReduceLROnPlateau déjà actif ; envisager LR cyclique."
        )
    else:
        recs["learning_rate"] = "✅  LR correct, convergence visible."

    # — Dropout / Overfitting —
    if gap > 0.25:
        recs["dropout"] = (
            f"Overfitting fort (gap train-val={gap:.2f}). "
            "Augmenter Dropout de +0.1 sur chaque couche Dense."
        )
        recs["augmentation"] = "Passer à augmentation 'aggressive'."
    elif gap > 0.12:
        recs["dropout"] = (
            f"Overfitting modéré (gap={gap:.2f}). "
            "Légère hausse du Dropout (+0.05) recommandée."
        )
        recs["augmentation"] = "Augmentation 'medium' → 'aggressive' si persistant."
    else:
        recs["dropout"] = f" Pas d'overfitting significatif (gap={gap:.2f})."
        recs["augmentation"] = "Augmentation actuelle suffisante."

    # — Epochs —
    if n >= EPOCHS_FULL:
        recs["epochs"] = (
            "EarlyStopping non déclenché : augmenter patience ou epochs max."
        )
    else:
        recs["epochs"] = f"Arrêt naturel à l'epoch {n} (EarlyStopping)."

    # — Top-5 —
    if "val_top5" in h:
        top5_final = h["val_top5"][-1]
        recs["top5"] = (
            f"Top-5 val = {top5_final:.4f} (objectif >0.80)"
            if top5_final > 0.80
            else f"Top-5 val = {top5_final:.4f} < 0.80 — architecture insuffisante."
        )

    return recs


# ══════════════════════════════════════════════════════════════
# 4. COMPARAISON MID-TRAINING (epoch 30)
# ══════════════════════════════════════════════════════════════

def mid_training_comparison(results: dict):
    """Tableau comparatif à l'epoch 30 (ou dernière disponible)."""
    print(f"\n{'═'*65}")
    print("  COMPARAISON MID-TRAINING  (epoch ≈ 30)")
    print(f"{'═'*65}")
    print(f"  {'Modèle':<22} {'Val Acc':>8} {'Val Loss':>9} {'Top-5':>7} {'Gap':>7}  Epoch")
    print(f"  {'-'*62}")

    rows = []
    for name, (history, _) in results.items():
        h  = history.history
        ep = min(30, len(h["val_accuracy"])) - 1
        va = h["val_accuracy"][ep]
        vl = h["val_loss"][ep]
        ta = h["accuracy"][ep]
        t5 = h["val_top5"][ep] if "val_top5" in h else float("nan")
        rows.append((name, va, vl, t5, ta - va, ep + 1))

    rows.sort(key=lambda x: x[1], reverse=True)
    for i, (name, va, vl, t5, gap, ep) in enumerate(rows):
        medal = ["🥇", "🥈", "🥉", "  "][min(i, 3)]
        t5_str = f"{t5:.4f}" if not np.isnan(t5) else "  N/A "
        print(f"  {medal} {name:<20} {va:>8.4f} {vl:>9.4f} {t5_str:>7} {gap:>7.4f}  ep{ep}")

    print(f"{'═'*65}\n")
    return rows


# ══════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════

COLORS = {
    "CNN Simple"    : "#4C9BE8",
    "CNN Profond"   : "#A67CF5",
    "MobileNetV2"   : "#5CC98B",
    "EfficientNetB0": "#F0883E",
}


def plot_individual(name: str, history):
    """Courbes loss / accuracy / top-5 pour un modèle."""
    h   = history.history
    has_top5 = "val_top5" in h
    ncols = 3 if has_top5 else 2

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    fig.suptitle(f"Courbes d'entraînement — {name}", fontsize=13, fontweight="bold")

    c = COLORS.get(name, "#888")

    axes[0].plot(h["loss"],     color=c,   label="Train")
    axes[0].plot(h["val_loss"], color=c, linestyle="--", label="Val")
    axes[0].set_title("Loss");  axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(h["accuracy"],     color=c,   label="Train")
    axes[1].plot(h["val_accuracy"], color=c, linestyle="--", label="Val")
    axes[1].axhline(0.40, color="gray", linestyle=":", linewidth=1, label="Objectif 40%")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend()

    if has_top5:
        axes[2].plot(h.get("top5", h.get("top5_accuracy", [])),     color=c,   label="Train")
        axes[2].plot(h["val_top5"], color=c, linestyle="--", label="Val")
        axes[2].axhline(0.80, color="gray", linestyle=":", linewidth=1, label="Objectif 80%")
        axes[2].set_title("Top-5 Accuracy"); axes[2].set_xlabel("Epoch")
        axes[2].legend()

    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, slug, "curves.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Courbes {name} → {path}")


def plot_comparison(results: dict):
    """Graphe comparatif val_accuracy pour tous les modèles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison globale — toutes architectures", fontsize=13, fontweight="bold")

    for name, (history, _) in results.items():
        c  = COLORS.get(name, "#888")
        h  = history.history
        axes[0].plot(h["val_accuracy"], color=c, label=name)
        axes[1].plot(h["val_loss"],     color=c, label=name)

    axes[0].axhline(0.40, color="gray", linestyle=":", label="Objectif 40%")
    axes[0].set_title("Val Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Val Loss");     axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparaison globale → {path}")


# ══════════════════════════════════════════════════════════════
# 6. ÉVALUATION FINALE SUR LE TEST SET
# ══════════════════════════════════════════════════════════════

def evaluate_on_test(results: dict, X_test, y_test_ohe):
    """Évalue chaque modèle sur le jeu de test et affiche le bilan."""
    print(f"\n{'═'*60}")
    print("  ÉVALUATION FINALE — TEST SET")
    print(f"{'═'*60}")
    print(f"  {'Modèle':<22} {'Loss':>7} {'Accuracy':>10} {'Top-5':>8}")
    print(f"  {'-'*57}")

    final = {}
    for name, (_, model) in results.items():
        out = model.evaluate(X_test, y_test_ohe, verbose=0)
        # out = [loss, accuracy] ou [loss, accuracy, top5]
        loss = out[0]; acc = out[1]
        top5 = out[2] if len(out) > 2 else float("nan")
        final[name] = {"test_loss": loss, "test_accuracy": acc, "test_top5": top5}

        t5_str = f"{top5:.4f}" if not np.isnan(top5) else "  N/A"
        ok_acc  = "✅" if acc  >= 0.40 else "❌"
        ok_top5 = "✅" if top5 >= 0.80 else "❌"
        print(f"  {name:<22} {loss:>7.4f} {acc:>8.4f} {ok_acc}  {t5_str} {ok_top5}")

    path = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n Résultats finaux → {path}")
    print(f"{'═'*60}\n")
    return final


# ══════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def run_phase4(augmentation_level="medium"):
    """
    Point d'entrée unique.
    Enchaîne : données → entraînement → tuning report → mid-training → plots → eval.
    """
    # ── Données ─────────────────────────────────────────────
    X_train, y_train_ohe, X_test, y_test_ohe = load_data()

    # ── Entraînement ────────────────────────────────────────
    results = train_all_models(X_train, y_train_ohe, augmentation_level)

    # ── Tuning adaptatif ────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  TUNING ADAPTATIF — RECOMMANDATIONS PAR MODÈLE")
    print(f"{'═'*65}")
    for name, (history, _) in results.items():
        print(f"\n  🔧  {name}")
        recs = adaptive_tuning_report(name, history)
        for key, msg in recs.items():
            print(f"     [{key}] {msg}")

    # ── Comparaison mid-training ─────────────────────────────
    mid_training_comparison(results)

    # ── Visualisations ───────────────────────────────────────
    for name, (history, _) in results.items():
        plot_individual(name, history)
    plot_comparison(results)

    # ── Évaluation finale ────────────────────────────────────
    evaluate_on_test(results, X_test, y_test_ohe)

    print(f"Phase 4 terminée. Tous les fichiers sont dans → {OUTPUT_DIR}/")
    print(f"""
Structure de sortie :
  {OUTPUT_DIR}/
  ├── cnn_simple/
  │   ├── best_model.keras
  │   ├── history.json
  │   └── curves.png
  ├── cnn_profond/        (idem)
  ├── mobilenetv2/        (idem)
  ├── efficientnetb0/     (idem)
  ├── comparison.png
  └── final_results.json
""")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase4(augmentation_level="medium")
    # Niveaux disponibles : "light" | "medium" | "aggressive"