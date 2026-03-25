# ⚡ PowerCast : Prévision de la Demande Énergétique

![PowerCast Banner](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge)
![JAX](https://img.shields.io/badge/JAX-Powered-blue?style=for-the-badge&logo=google)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)

**PowerCast** est un projet professionnel (Portfolio) de prévision de la demande énergétique nationale (France) basé sur les données publiques ouvertes de réseau de transport d'électricité (RTE - ODRE).

L'objectif de ce projet est de démontrer l'implémentation "from-scratch" d'une architecture **Transformer Encoder-only** en utilisant le framework haute performance **JAX/Flax**, ainsi que l'explicabilité du modèle via **SHAP**.

## 🌟 Fonctionnalités
- **Pipeline de données complet** : Téléchargement, nettoyage, et interpolation des données manquantes sur les séries temporelles RTE (pas de 30 minutes).
- **Baselines robustes** : Implémentation de modèles de référence (Prophet et LSTM PyTorch) pour évaluation rigoureuse de la valeur ajoutée du Transformer.
- **Deep Learning JAX/Flax** : Architecture auto-régressive avec `PositionalEncoding` et `MultiHeadDotProductAttention` entièrement compilée via `jax.jit`.
- **Explicabilité (XAI)** : Intégration de `shap.KernelExplainer` pour interpréter les fenêtres de temps passées qui influencent le plus la prédiction.
- **Dashboard Interactif** : Interface utilisateur en temps réel construite avec Streamlit, permettant d'observer l'inférence du modèle sur de nouvelles données.

## 📁 Architecture du Projet
```text
PowerCast/
├── app/
│   └── app.py                   # Application Streamlit principale
├── data/
│   ├── processed/               # Données nettoyées
│   └── results/                 # Poids du modèle et graphiques
├── notebooks/
│   ├── 01_EDA_RTE.ipynb         # Analyse exploratoire
│   └── 02_Transformer_Training.ipynb # Notebook Colab pour l'entraînement GPU
├── src/
│   ├── data/
│   │   ├── download_rte_data.py # Script d'acquisition
│   │   └── preprocess.py        # Script de nettoyage et interpolation
│   └── models/
│       ├── baselines.py         # Baseline Prophet
│       ├── lstm_baseline.py     # Baseline LSTM (PyTorch)
│       └── transformer_jax.py   # Modèle Transformer (JAX/Flax)
└── requirements.txt             # Dépendances du projet
```

## 🚀 Utilisation et Déploiement

Le code d'entraînement intensif est conçu pour être lancé sur **Google Colab** (GPU T4) via le notebook fourni `notebooks/02_Transformer_Training.ipynb`. Les poids du modèle (`transformer_weights.msgpack`) sont ensuite chargés par l'application pour l'inférence.

**Lancer le Dashboard Localement :**
*(Nécessite un environnement compatible avec JAX, de préférence Linux ou WSL2. Sur Windows natif, privilégier le déploiement Cloud).*
```bash
pip install -r requirements.txt
# S'assurer d'avoir les données dans data/processed/ (télécharger depuis app.py si absents)
streamlit run app/app.py
```

**Déploiement Cloud (Streamlit Community Cloud) :**
L'application est configurée pour être déployée instantanément. Poussez le code sur GitHub, liez le dépôt à Streamlit Cloud, et l'environnement serveur (incluant JAX et SHAP) sera automatiquement configuré.

## 📊 Résultats
- **Baseline Prophet** : MAPE ~ 21.50% (Saisonnalité pure)
- **Baseline LSTM (1-step ahead)** : MAPE ~ 1.18% (Auto-régressif court terme)
- **Transformer Encoder-only** : Convergence rapide (accélération XLA) offrant une puissance prédictive multi-têtes expliquable structurellement.

## 🔗 Auteur
Développé dans le cadre d'un master **Mathématiques Appliquées**, spécialité Séries Temporelles, Machine Learning & Transformers.

- Feature update 1

- Feature update 2

- Feature update 3
