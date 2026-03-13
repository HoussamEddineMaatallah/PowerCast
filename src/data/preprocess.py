import pandas as pd
import numpy as np
import os

def preprocess_rte_data(input_path="data/raw/eco2mix_national.csv", output_path="data/processed/eco2mix_cleaned.csv"):
    print(f"Chargement des données depuis {input_path}...")
    df = pd.read_csv(input_path, sep=';', low_memory=False)
    
    # Selection of relevant columns
    cols_to_keep = ['Date et Heure', 'Consommation (MW)']
    df = df[cols_to_keep].copy()
    
    # Renaming
    df.rename(columns={'Date et Heure': 'datetime', 'Consommation (MW)': 'consumption'}, inplace=True)
    
    print(f"Taille initiale: {df.shape}")
    
    # Drop rows without datetime
    df.dropna(subset=['datetime'], inplace=True)
    
    # Convert to datetime UTC
    print("Conversion des dates...")
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df.sort_values('datetime', inplace=True)
    
    # Handle duplicates and missing steps (resampling at 30min)
    df.set_index('datetime', inplace=True)
    # Remove duplicates that could arise from daylight saving time
    df = df[~df.index.duplicated(keep='first')]
    
    print("Rééchantillonnage à 30 minutes (demi-horaire)...")
    # '30min' frequency
    df = df.resample('30min').asfreq()
    
    missing_cons = df['consumption'].isna().sum()
    if missing_cons > 0:
        print(f"Interpolation de {missing_cons} valeurs manquantes...")
        df['consumption'] = df['consumption'].interpolate(method='time')
        
    df.reset_index(inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Données nettoyées sauvegardées dans {output_path}")
    print(f"Taille finale: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    preprocess_rte_data()
