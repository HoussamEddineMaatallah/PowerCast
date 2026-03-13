import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import Dataset, DataLoader
import torch

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

# 1. Positional Encoding
class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    @nn.compact
    def __call__(self, x):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        seq_len = x.shape[1]
        pe_slice = jnp.array(pe[:seq_len, :])
        return x + pe_slice

# 2. Encoder Block
class TransformerEncoderBlock(nn.Module):
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        # Attention
        x = nn.LayerNorm()(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            qkv_features=inputs.shape[-1] // self.num_heads
        )(x, x, deterministic=deterministic)
        x = x + inputs
        
        # MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(inputs.shape[-1])(y)
        y = y + x
        return y

# 3. Full Transformer Model
class TimeSeriesTransformer(nn.Module):
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 64
    dropout_rate: float = 0.1
    pred_len: int = 1

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Feature embedding: (batch, seq_len, 1) -> (batch, seq_len, d_model)
        x = nn.Dense(self.d_model)(x)
        x = PositionalEncoding(d_model=self.d_model)(x)
        
        # Encoder blocks
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate
            )(x, deterministic=deterministic)
            
        # Global Average Pooling (or take last token)
        # Here we take the last token representation
        x = x[:, -1, :]
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.pred_len)(x)
        return x

def run_transformer_jax(data_path="data/processed/eco2mix_cleaned.csv"):
    print("Mise en place de la baseline Transformer (JAX/Flax)...")
    
    # 1. Prepare Data
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
    
    seq_length = 168
    pred_length = 1
    
    test_context = np.concatenate([train_values[-seq_length:], test_values])
    test_scaled = scaler.transform(test_context.reshape(-1, 1)).flatten()
    
    train_dataset = TimeSeriesDataset(train_scaled, seq_length, pred_length)
    test_dataset = TimeSeriesDataset(test_scaled, seq_length, pred_length)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Init Model & TrainState
    model = TimeSeriesTransformer()
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Dummy input
    dummy_x = jnp.ones((1, seq_length, 1))
    variables = model.init(init_rng, dummy_x, deterministic=True)
    
    tx = optax.adam(learning_rate=0.005)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

    # 3. Define Loss & Step functions
    @jax.jit
    def train_step(state, batch_x, batch_y, dropout_rng):
        def loss_fn(params):
            preds = state.apply_fn({'params': params}, batch_x, deterministic=False, rngs={'dropout': dropout_rng})
            loss = jnp.mean(optax.l2_loss(preds, batch_y))
            return loss, preds
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, preds), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(state, batch_x):
        return state.apply_fn({'params': state.params}, batch_x, deterministic=True)

    # 4. Training Loop
    epochs = 3
    print("Début de l'entraînement JAX...")
    for epoch in range(epochs):
        rng, dropout_rng = jax.random.split(rng)
        
        total_loss = 0.0
        start_time = time.time()
        for x_batch, y_batch in train_loader:
            x_b = jnp.array(x_batch.numpy())
            y_b = jnp.array(y_batch.numpy())
            
            rng, dropout_batch_rng = jax.random.split(rng)
            state, loss = train_step(state, x_b, y_b, dropout_batch_rng)
            total_loss += loss
            
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - {elapsed:.1f}s")

    # 5. Evaluation
    print("Évaluation sur le jeu de test...")
    predictions = []
    actuals = []
    
    for x_batch, y_batch in test_loader:
        x_b = jnp.array(x_batch.numpy())
        preds = eval_step(state, x_b)
        predictions.append(np.array(preds))
        actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    predictions = scaler.inverse_transform(predictions).flatten()
    actuals = scaler.inverse_transform(actuals).flatten()
    
    metrics = {
        'RMSE': root_mean_squared_error(actuals, predictions),
        'MAE': mean_absolute_error(actuals, predictions),
        'MAPE': mean_absolute_percentage_error(actuals, predictions) * 100
    }
    
    print("\n--- Métriques de performance (Transformer JAX 1-step ahead) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    os.makedirs("data/results", exist_ok=True)
    plt.figure(figsize=(15, 6))
    time_axis = df['datetime'][~train_mask].values
    plt.plot(time_axis, actuals, label='Vérité terrain')
    plt.plot(time_axis, predictions, label='Transformer (Prédiction)', alpha=0.7)
    plt.title(f"Prédiction Transformer - Derniers 30 jours (MAPE: {metrics['MAPE']:.2f}%)")
    plt.legend()
    plt.savefig("data/results/transformer_forecast.png")
    print("Graphique sauvegardé dans data/results/transformer_forecast.png")

if __name__ == "__main__":
    run_transformer_jax()
