TP Groupe : Classification d'Images CIFAR-100
Équipe: 3-4 personnes

Objectif
Construire un système complet de classification d'images sur le dataset CIFAR-100. Le projet teste plusieurs architectures deep learning : CNN personnalisés, transfer learning, fine-tuning, data augmentation, et analyse approfondie de la robustesse sur 100 classes.

Dataset Réel : CIFAR-100
Source: Directement intégré dans TensorFlow

from tensorflow.keras.datasets import cifar100
(X_train, y_train), (X_test, y_test) = cifar10l0.load_data(label_mode='fine')
# 'fine' = 100 classes, 'coarse' = 20 super-classes
Caractéristiques:

60,000 images RGB 32x32
100 classes fine-grained (pomme, aquarium, bébé, ours, castor, lit, abeille, scarabée, vélo, bouteille, bol, garçon, pont, bus, papillon, chameau, canette, château, chenille, bétail, chaise, chimpanzé, horloge, nuage, cafard, canapé, crabe, crocodile, tasse, dinosaure, dauphin, éléphant, plat, forêt, renard, fille, hamster, maison, kangourou, clavier, lampe, tondeuse, léopard, lion, lézard, homard, homme, érable, moto, montagne, souris, champignon, chêne, orange, orchidée, loutre, palmier, poire, pickup, pin, plaine, plat, pavot, porc-épic, opossum, lapin, raton laveur, raie, route, fusée, rose, mer, phoque, requin, musaraigne, mouffette, gratte-ciel, escargot, serpent, araignée, écureuil, streetcar, tournesol, patate douce, table, tank, téléphone, télévision, train, truite, tulipe, tortue, garde-robe, baleine, saule, loup, femme, ver)
20 super-classes (aquatic mammals, fish, flowers, food containers, fruit and vegetables, household electrical devices, household furniture, insects, large carnivores, large man-made outdoor things, large natural outdoor scenes, large omnivores and herbivores, medium-sized mammals, non-insect invertebrates, people, reptiles, small mammals, trees, vehicles 1, vehicles 2)
600 images par classe (vs 6,000 pour CIFAR-10)
Train set: 50,000 images
Test set: 10,000 images
Données réelles : objets variés en conditions naturelles
Classes équilibrées
Challenges réels (100 classes):

Images petites (32x32, information limitée)
100 classes = beaucoup plus difficile que CIFAR-10
Peu d'exemples par classe (600 vs 6,000)
Certaines classes très similaires (pomme vs orange, loup vs chien)
Variabilité d'apparence intra-classe
Arrière-plans complexes
Hiérarchie super-classes/classes à exploiter
Contexte Réel
Vous travaillez pour une entreprise d'intelligence artificielle développant un système de reconnaissance visuelle avancée. Objectif:

Classifier automatiquement 100 catégories d'objets détectés
Gérer la complexité de classes très similaires (fine-grained classification)
Exploiter la hiérarchie des super-classes pour améliorer la robustesse
Système temps réel (faible latence)
Robustesse aux conditions de déploiement avec peu de données par classe
Répartition des Rôles
Rôle	Responsabilités
Lead Data	Exploration, augmentation, préparation
Lead DL	Architecture design, entraînement, tuning
Lead Analyse	Évaluation, robustesse, interprétabilité
Lead Synthesis	Rapport, documentation, conclusions
Phase 1 : Exploration des Images
Analyser en détail le dataset CIFAR-100.

Tâches:

Charger et inspecter

Dimensions: (32, 32, 3)
Nombre images par classe (600 seulement!)
Range des valeurs pixels
Charger labels fine (100 classes) et coarse (20 super-classes)
Visualisation

Afficher 5-10 images aléatoires pour 20 classes sélectionnées
Montrer variation intra-classe
Montrer images faciles vs difficiles
Identifier confusions possibles (loup/chien, pomme/orange)
Visualiser la hiérarchie super-classes → classes
Statistiques images

Distribution luminosité
Distribution contraste
Distribution couleur (RGB stats)
Images corrompues ou bizarres?
Analyse classes confondues

Animaux similaires : loup vs chien vs renard
Fruits : pomme vs orange vs poire
Véhicules : bus vs train vs streetcar
Arbres : chêne vs érable vs saule vs pin
Analyse des super-classes

Distribution des 100 classes dans les 20 super-classes
Classes les plus difficiles par super-classe
Stratégie : prédire d'abord super-classe puis classe?
Planification stratégie

Data augmentation essentielle (peu de données)
Quelle taille cible? (32x32 ou upscale pour transfer learning?)
Normalisation : comment?
Transfer learning obligatoire (600 images/classe insuffisant)
Phase 2 : Préparation & Augmentation
Préparer les données pour deep learning.

Tâches:

Normalisation
```python # Option 1: ImageNet normalization (si transfer learning) mean = [0.485, 0.456, 0.406] std = [0.229, 0.224, 0.225]

# Option 2: Simple scaling X_train = X_train / 255.0 X_test = X_test / 255.0 ```

One-hot encoding
```python from tensorflow.keras.utils import to_categorical y_train_fine = to_categorical(y_train, 100) # 100 classes y_test_fine = to_categorical(y_test, 100)

# Optionnel : super-classes y_train_coarse = to_categorical(y_train_coarse, 20) # 20 super-classes ```

Data augmentation (CRITIQUE pour 600 images/classe)

Rotation: ±20 degrés
Zoom: ±25%
Shift: ±15%
Flip horizontal: activé
Brightness: ±20%
Cutout/Random Erasing (optionnel mais recommandé)
Mixup (optionnel avancé)
Validation augmentation

Afficher images augmentées
Vérifier labels inchangés
Mesurer impact training
Phase 3 : Architectures de Deep Learning
Tester 4 approches différentes (attention : 100 classes au lieu de 10!).

Architecture 1 : CNN Personnalisé Simple

Sequential([
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
Architecture 2 : CNN Plus Profond (ResNet-like)

# Avec residual connections
# 6+ couches conv
# Skip connections tous les 2 blocs
# Batch normalization systématique
Architecture 3 : Transfer Learning - MobileNetV2

# Upscale 32x32 → 96x96 ou 224x224 pour transfer learning
base_model = MobileNetV2(
    input_shape=(96, 96, 3),  # ou (224, 224, 3)
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    layers.Resizing(96, 96),  # Redimensionner à l'entrée
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax')  # 100 classes!
])
Architecture 4 : Transfer Learning + Fine-tuning - EfficientNetB0

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),  # Redimensionner CIFAR-100
    include_top=False,
    weights='imagenet'
)

# Fine-tuning : dégeler couches finales
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Dégeler plus de couches
    layer.trainable = False

model = Sequential([
    layers.Resizing(224, 224),  # Redimensionner à l'entrée
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax')  # 100 classes!
])
Architecture 5 (Bonus) : Modèle Hiérarchique

# Prédire d'abord la super-classe (20), puis la classe fine (100)
# Deux branches de sortie
# Peut améliorer la performance en exploitant la hiérarchie
Hyperparamètres communs:

Batch size: 64
Learning rate: 0.001
Epochs: 100 (avec early stopping)
Optimizer: Adam
Loss: categorical_crossentropy
Callbacks obligatoires:

EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
Phase 4 : Entraînement & Optimisation
Entraîner chaque architecture.

Tâches:

Entraînement

Entraîner chaque modèle
Monitor train/val loss et accuracy
Sauvegarder checkpoints
Exporter historique d'entraînement
Tuning adaptatif

Ajuster learning rate selon courbes
Modifier dropout si overfitting excessif
Augmenter epochs si amélioration continue
Data augmentation plus/moins agressive?
Comparison mid-training

Après 30 epochs, quel modèle en tête?
Converge vite / lentement?
Overfitting visible?
Performance attendue : 40-60% accuracy (100 classes est difficile!)
Top-5 accuracy : doit être >80%
Phase 5 : Évaluation Complète
Évaluer rigoureusement chaque modèle.

Tâches:

Métriques globales

Accuracy sur test set
Precision/Recall/F1 par classe
Top-3 accuracy
Confusion matrix
Analyse détaillée par classe

Classes faciles (accuracy > 95%)
Classes difficiles (accuracy < 80%)
Confusions courantes
Images mal classifiées : pourquoi?
Comparaison architectures

python # Tableau récapitulatif (100 classes): | Architecture | Top-1 Acc | Top-5 Acc | F1 | Speed (ms) | |---|---|---|---|---| | CNN Simple | 42% | 68% | 0.40 | 2ms | | CNN Profond | 48% | 74% | 0.46 | 5ms | | MobileNetV2 | 52% | 78% | 0.50 | 12ms | | EfficientNetB0 | 58% | 83% | 0.56 | 30ms | | Hiérarchique (bonus) | 55% | 80% | 0.53 | 8ms |

Note : 100 classes rend la tâche beaucoup plus difficile! - Accuracy typique CIFAR-100 : 50-65% - Top-5 accuracy : 75-85%

Trade-offs
Accuracy vs vitesse (deployment trade-off)
Complexité vs performance
Meilleur modèle pour production?
Phase 6 : Robustesse
Tester comment le modèle se comporte en conditions réelles.

Tâches:

Augmentation synthétique du test set

Ajouter bruit Gaussian
Ajouter bruit Poisson
Rotation ±10 degrés
Occlusion (masquer 25% de l'image)
Mesurer dégradation

python for noise_level in [0.1, 0.2, 0.3]: perturbed = add_noise(X_test, noise_level) accuracy_noisy = evaluate(model, perturbed) print(f"Accuracy with {noise_level} noise: {accuracy_noisy}")

Analyse adversariale simple
Générer small perturbations
Tester robustesse
Identifier vulnérabilités
Phase 7 : Interprétabilité (Optionnel)
Comprendre ce que le modèle apprend.

Tâches:

Feature maps

Visualiser activations des couches intermédiaires
Voir patterns détectés
Comparaison architectures
Grad-CAM (si temps)

Implémenter Grad-CAM
Générer heatmaps pour 20 images
Vérifier que modèle regarde régions pertinentes
Chat : regarde oreilles/yeux?
Livrables Finaux
Notebook Jupyter

Exécutable
Code commenté
Résultats visibles
Modèles

4 modèles entraînés (.h5 ou .keras)
Meilleur modèle spécifié
Poids sauvegardés
Rapport

Complet et structuré
Visualisations
Données chiffrées
Visualisations

Courbes d'apprentissage
Confusion matrices
Feature maps (optionnel)
Conseils Pratiques
Commencez avec architecture simple (CNN Simple)
Progressez graduellement (CNN Profond → Transfer)
Sauvegarder modèles fréquemment
Monitor GPU memory (si disponible)
Utilisez random seeds (reproducibilité)
Pair programming sur architecture design
Documentez décisions au fur et à mesure