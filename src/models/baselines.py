import pandas as pd
import numpy as np
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prophet import Prophet
import matplotlib.pyplot as plt

def evaluate_metrics(y_true, y_pred):
    return {
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

def run_prophet_baseline(data_path="data/processed/eco2mix_cleaned.csv"):
    print("Chargement des données pour Prophet...")
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Prophet requires 'ds' and 'y' columns
    df_prophet = df[['datetime', 'consumption']].rename(columns={'datetime': 'ds', 'consumption': 'y'})
    
    # Remove timezone for Prophet
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    # Train/Test Split (Test = Last 30 days)
    last_date = df_prophet['ds'].max()
    train_end = last_date - pd.Timedelta(days=30)
    train_start = train_end - pd.Timedelta(days=365 * 2) # Train on 2 years to keep it fast but robust
    
    print(f"Période d'entraînement: {train_start} -> {train_end}")
    print(f"Période de test: {train_end} -> {last_date}")
    
    train_df = df_prophet[(df_prophet['ds'] >= train_start) & (df_prophet['ds'] < train_end)]
    test_df = df_prophet[df_prophet['ds'] >= train_end]
    
    print(f"Taille Train: {len(train_df)}, Taille Test: {len(test_df)}")
    
    # Initialize and train Prophet
    print("Entraînement de Prophet en cours...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(train_df)
    
    # Forecast
    print("Génération de la prédiction pour le set de Test...")
    forecast = model.predict(test_df[['ds']])
    
    # Evaluate
    metrics = evaluate_metrics(test_df['y'].values, forecast['yhat'].values)
    print("\n--- Métriques de performance (Prophet) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    print(f"MAPE = {metrics['MAPE']:.2f}% (Interprétation: Se trompe en moyenne de {metrics['MAPE']:.2f}%)")
    
    # Save a plot
    os.makedirs("data/results", exist_ok=True)
    plt.figure(figsize=(15, 6))
    plt.plot(test_df['ds'], test_df['y'], label='Vérité terrain')
    plt.plot(forecast['ds'], forecast['yhat'], label='Prophet (Prédiction)', alpha=0.7)
    plt.title(f"Prédiction Prophet - Derniers 30 jours (MAPE: {metrics['MAPE']:.2f}%)")
    plt.legend()
    plt.savefig("data/results/prophet_forecast.png")
    print("Graphique sauvegardé dans data/results/prophet_forecast.png")

if __name__ == "__main__":
    run_prophet_baseline()
