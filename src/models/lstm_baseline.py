import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import time

def evaluate_metrics(y_true, y_pred):
    return {
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def run_lstm_baseline(data_path="data/processed/eco2mix_cleaned.csv"):
    print("Mise en place de la baseline LSTM...")
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    last_date = df['datetime'].max()
    test_start = last_date - pd.Timedelta(days=30)
    train_start = test_start - pd.Timedelta(days=365 * 2)
    
    df = df[(df['datetime'] >= train_start) & (df['datetime'] <= last_date)].copy()
    
    values = df['consumption'].values
    train_mask = df['datetime'] < test_start
    train_values = values[train_mask]
    test_values = values[~train_mask]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
    
    seq_length = 168 # 3.5 days (30min interval)
    pred_length = 1
    
    test_context = np.concatenate([train_values[-seq_length:], test_values])
    test_scaled = scaler.transform(test_context.reshape(-1, 1)).flatten()
    
    train_dataset = TimeSeriesDataset(train_scaled, seq_length, pred_length)
    test_dataset = TimeSeriesDataset(test_scaled, seq_length, pred_length)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")
    model = LSTMModel(input_size=1, hidden_size=32, num_layers=1, output_size=pred_length).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 3
    print("Début de l'entraînement...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - {elapsed:.1f}s")
        
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            out = model(x_batch).cpu().numpy()
            predictions.append(out)
            actuals.append(y_batch.numpy())
            
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    predictions = scaler.inverse_transform(predictions).flatten()
    actuals = scaler.inverse_transform(actuals).flatten()
    
    metrics = evaluate_metrics(actuals, predictions)
    print("\n--- Métriques de performance (LSTM 1-step ahead) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    
    os.makedirs("data/results", exist_ok=True)
    plt.figure(figsize=(15, 6))
    time_axis = df['datetime'][~train_mask].values
    plt.plot(time_axis, actuals, label='Vérité terrain')
    plt.plot(time_axis, predictions, label='LSTM (Prédiction)', alpha=0.7)
    plt.title(f"Prédiction LSTM - Derniers 30 jours (MAPE: {metrics['MAPE']:.2f}%)")
    plt.legend()
    plt.savefig("data/results/lstm_forecast.png")
    print("Graphique sauvegardé dans data/results/lstm_forecast.png")

if __name__ == "__main__":
    run_lstm_baseline()
