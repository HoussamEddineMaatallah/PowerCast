# Plan d'implémentation : Prévision de la demande énergétique (PowerCast)

Ce document décrit la stratégie technique pour implémenter un modèle Transformer prédisant la demande électrique française, en utilisant JAX/Flax. L'objectif est de réaliser un projet "from-scratch", de l'acquisition des données à l'inférence via une interface Streamlit.

## User Review Required
> [!IMPORTANT]
> Les points suivants nécessitent ton avis avant de commencer le développement réel :
> 1. **Acquisition des données :** Télécharger des fichiers CSV statiques depuis l'Open Data RTE est le plus direct. L'utilisation de l'API requiert la création d'un compte développeur gratuit pour générer un jeton. Préfères-tu automatiser via l'API, ou démarrer plus simplement avec des CSV ?
> 2. **Complexité Initiale :** Veux-tu qu'on commence par une version strictement univariée (seulement la consommation) pour valider l'architecture du Transformer, avant d'ajouter les features exogènes (Météo, Jours Fériés) dans un second temps (Jalon 4) ?

## Proposed Changes

La structure suivante sera mise en place dans ton répertoire de travail local (`C:\Users\houss\OneDrive\Desktop\PowerCast`) :

### Architecture du Répertoire

```text
PowerCast/
├── data/               # Données brutes et traitées (ignoré par git)
├── notebooks/          # Notebooks Jupyter/Colab pour EDA et prototypage
├── src/                # Code source Python (modules réutilisables)
│   ├── data/           # Scripts de téléchargement et prétraitement
│   ├── models/         # Architectures (Baselines, Transformer_JAX)
│   ├── training/       # Boucles d'entraînement
│   └── explain/        # Code pour SHAP et Attention
├── app/                # Code de l'application Streamlit
├── requirements.txt    # Dépendances du projet
└── README.md
```

### Modélisation & Pipeline Technique
- **Baselines** : `Prophet` (robustesse aux saisonnalités complexes) et `LSTM` (sur PyTorch ou JAX, pour comparer un réseau récurrent classique face au mécanisme d'attention).
- **Transformer (JAX/Flax)** :
  - **Entrée** : Fenêtre glissante de longueur cible (ex: contexte de 7 jours = 168 heures).
  - **Encodeur** : Multi-Head Self Attention (Architecture Encoder-only car nous faisons de la prédiction directe d'une séquence cible, souvent plus robuste et rapide pour des séries temporelles).
  - **Avantage JAX** : Compilation `jax.jit` pour des vitesses d'entraînement considérablement accélérées sur les GPU T4 gratuits de Google Colab.

### Déploiement et Interprétabilité
- **Interface** : Streamlit pour sa simplicité et son intégration Python-native.
- **Visualisations** : Cartes de chaleur (heatmaps) d'attention et intégration de `SHAP values` pour l'explicabilité du modèle.

## Verification Plan

### Automated/Manual Tests
- **Split Temporel des Données** : Séparation chronologique stricte pour éviter la fuite de données (data leakage). Exemple : entraînement 2015-2021, validation 2022 et test 2023.
- **Métriques d'évaluation** : 
  - RMSE (pour pénaliser fortement les erreurs sur les pics de consommation extrême, typiques en hiver).
  - MAE (interprétable directement en MWh).
  - MAPE (pourcentage d'erreur relatif, incontournable pour la communication).
- **Test d'interface (Manuel)** : Lancement en local avec `streamlit run app/app.py` pour valider l'interaction : sélection d'une date du subset de test, appel aux poids du modèle, affichage superposé prévision vs vérité terrain.
