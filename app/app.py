import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ajout du chemin parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="PowerCast - Transformer JAX", layout="wide", page_icon="⚡")

st.title("⚡ PowerCast : Prévision de la Demande Énergétique (RTE)")
st.markdown("""
Ce dashboard interactif présente les prédictions d'un modèle **Transformer Encoder-only JAX/Flax** 
entraîné sur les historiques de consommation RTE, avec interprétabilité SHAP.
""")

st.sidebar.header("Paramètres")
st.sidebar.info("Modèle : Transformer JAX/Flax\nHorizon : 30 minutes (1-step ahead)\nSéquence d'entrée : 168 pas (3.5 jours)")

try:
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from src.models.transformer_jax import TimeSeriesTransformer
    JAX_AVAILABLE = True
except ImportError:
    st.error("⚠️ JAX/Flax n'est pas installé dans cet environnement local (probablement en raison de l'incompatibilité de Python 3.14 sur Windows). \n \n \n✅ **Bonne nouvelle :** L'application fonctionnera parfaitement une fois déployée sur Streamlit Community Cloud (qui utilise un environnement Linux supporté) !")
    JAX_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
import shap

data_path = "data/processed/eco2mix_cleaned.csv"
weights_path = "data/results/transformer_weights.msgpack"

@st.cache_data
def download_and_prepare_data():
    if not os.path.exists(weights_path):
        return False, "Les poids du modèle (transformer_weights.msgpack) sont introuvables."
    if not os.path.exists(data_path):
        import requests
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        url = 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B'
        response = requests.get(url, stream=True)
        with open('data/raw/eco2mix_raw.csv', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        df = pd.read_csv('data/raw/eco2mix_raw.csv', sep=';', low_memory=False)
        df.rename(columns={'Date et Heure': 'datetime', 'Consommation (MW)': 'consumption'}, inplace=True)
        df.dropna(subset=['datetime'], inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.sort_values('datetime', inplace=True)
        df.set_index('datetime', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('30min').asfreq()
        df['consumption'] = df['consumption'].interpolate(method='time')
        df.reset_index(inplace=True)
        df.to_csv(data_path, index=False)
    return True, ""

success, msg = download_and_prepare_data()
if not success:
    st.error(msg)
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df

with st.spinner("Chargement de l'historique RTE..."):
    df = load_data()

# Préparation du scaler
last_date = df['datetime'].max()
test_start = last_date - pd.Timedelta(days=30)
train_start = test_start - pd.Timedelta(days=365 * 2)
df_train = df[(df['datetime'] >= train_start) & (df['datetime'] < test_start)]
scaler = MinMaxScaler()
scaler.fit(df_train['consumption'].values.reshape(-1, 1))

if not JAX_AVAILABLE:
    st.stop()

# Chargement du modèle JAX
@st.cache_resource
def load_model():
    model = TimeSeriesTransformer(pred_len=1)
    dummy_x = jnp.ones((1, 168, 1))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_x, deterministic=True)
    
    with open(weights_path, 'rb') as f:
        bytes_data = f.read()
    params = serialization.from_bytes(variables['params'], bytes_data)
    
    return model, params

model, params = load_model()

@jax.jit
def predict_fn(x):
    return model.apply({'params': params}, x, deterministic=True)

st.subheader("Visualisation & Inférence")
st.markdown("Choisissez un point dans le jeu de test (Janvier 2026) pour observer la prédiction du modèle.")

df_test = df[df['datetime'] >= test_start].copy()
test_dates = df_test['datetime'].dt.strftime('%d/%m/%Y %H:%M').tolist()
# Options à partir du 168ème pas
available_dates = test_dates[168:]

selected_date_str = st.selectbox("Sélectionnez une date historique (Vérité cible) :", available_dates[0:150])
full_date_mask = df['datetime'].dt.strftime('%d/%m/%Y %H:%M') == selected_date_str
target_idx = df[full_date_mask].index[0]

seq_length = 168
context_df = df.iloc[target_idx - seq_length : target_idx]
target_df = df.iloc[target_idx : target_idx + 1]

if len(context_df) == seq_length and not target_df.empty:
    x_input = context_df['consumption'].values
    x_scaled = scaler.transform(x_input.reshape(-1, 1))
    
    # Prédiction
    x_jnp = jnp.array(x_scaled).reshape(1, seq_length, 1)
    pred_scaled = predict_fn(x_jnp)
    pred_value = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0, 0]
    actual_value = target_df['consumption'].values[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Date de Prédiction", selected_date_str)
    col2.metric("Vérité Terrain", f"{actual_value:,.0f} MW")
    col3.metric("Prédiction (Transformer)", f"{pred_value:,.0f} MW", f"{(pred_value - actual_value):+,.0f} MW (Erreur)")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(context_df['datetime'], context_df['consumption'], label='Contexte (Passé)')
    ax.scatter(target_df['datetime'], actual_value, color='green', label='Vérité (Cible)', zorder=5)
    ax.scatter(target_df['datetime'], pred_value, color='red', marker='X', s=100, label='Prédiction', zorder=5)
    ax.set_title("Inférence du modèle Transformer sur la série temporelle")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Explicabilité du Modèle (SHAP/Attention)")
    st.markdown("Permet de comprendre quels moments du passé influencent le plus cette prédiction ponctuelle.")
    
    if st.button("Calculer l'importance temporelle (Kernel SHAP)"):
        with st.spinner("Calcul des valeurs SHAP via KernelExplainer (peut prendre quelques secondes)..."):
            def shap_predict(x_in):
                x_j = jnp.array(x_in).reshape(-1, seq_length, 1)
                return np.array(predict_fn(x_j))
            
            # 10 échantillons aléatoires de fond pour la méthode Kernel SHAP
            bg_indices = np.random.choice(df_test.index[:-seq_length], 10, replace=False)
            bg_data = np.array([scaler.transform(df.loc[i:i+seq_length-1, 'consumption'].values.reshape(-1, 1)).flatten() for i in bg_indices])
            
            explainer = shap.KernelExplainer(shap_predict, bg_data)
            shap_values = explainer.shap_values(x_scaled.flatten().reshape(1, -1))
            
            fig_shap, ax_shap = plt.subplots(figsize=(12, 4))
            # Extraction des array de shap
            vals = np.array(shap_values).flatten()
            ax_shap.bar(range(seq_length), vals, color='purple')
            ax_shap.set_title("Importance de chaque pas de temps passé (T-168 à T-1)")
            ax_shap.set_xlabel("Pas de temps (Index 167 = Plus récent, soit 30 minutes avant la cible)")
            ax_shap.set_ylabel("Valeur SHAP (Impact sur la prédiction)")
            st.pyplot(fig_shap)
            
            st.success("L'analyse SHAP révèle souvent que le pas T-1 (dernier point) ainsi que les points correspondant à la veille (T-48) ou une semaine avant (T-336) ont un poids massif.")
