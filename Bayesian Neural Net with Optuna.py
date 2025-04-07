import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os
import json
from scipy.stats import norm

# -------------------------------
#  1. Device Config
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
#  2. Load and Prepare Dataset
# -------------------------------
df = pd.read_csv("/Users/charliemurray/Documents/all_cohesionless_data/merged_filtered_data.csv")

input_columns = ["gt_c", "carr_index", "dynamic_angle_of_repose", "rpm", "size", 'angle_of_repose', "3rd_order_cubic_term"]
output_columns = ["restitution", "sliding_friction", "rolling_friction"]

df = df.dropna(subset=input_columns, how='all')

X = df[input_columns].values.astype(np.float32)
Y = df[output_columns].values.astype(np.float32)

X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
    print("Warning: NaN values found!")
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)

dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# -------------------------------
#  3. Bayesian Neural Network
# -------------------------------
class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_mean = nn.Linear(hidden_dim, output_dim)
        self.out_logvar = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.out_mean.weight, mean=0, std=0.01)
        nn.init.normal_(self.out_logvar.weight, mean=0, std=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        mean = self.out_mean(x)
        log_var = self.out_logvar(x)
        var = torch.exp(log_var) + 1e-6
        return mean, var

# -------------------------------
#  4. Loss and Training Functions
# -------------------------------
def gaussian_nll(y_true, mean, var):
    return torch.mean(0.5 * torch.log(var) + 0.5 * ((y_true - mean)**2) / var)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        mean, var = model(xb)
        loss = gaussian_nll(yb, mean, var)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mean, var = model(xb)
            loss = gaussian_nll(yb, mean, var)
            total_loss += loss.item()
    return total_loss / len(loader)

# -------------------------------
#  5. Hyperparameter Tuning
# -------------------------------
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    epochs = trial.suggest_int('epochs', 50, 150, step=50)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.1, 0.5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BNN(X.shape[1], Y.shape[1], hidden_dim, dropout_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer)
        val_loss = eval_epoch(model, val_loader)

    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=150)

print(f"\nBest Hyperparameters: {study.best_params}")

# -------------------------------
#  6. Final Model Training
# -------------------------------
params = study.best_params
train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)

model = BNN(X.shape[1], Y.shape[1], params['hidden_dim'], params['dropout_prob']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

train_losses, val_losses = [], []
for epoch in range(params['epochs']):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = eval_epoch(model, val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# -------------------------------
#  7. Save Results
# -------------------------------
results_dir = "bayesian_neural_net/results"
os.makedirs(results_dir, exist_ok=True)

# Save learning curves
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("BNN Learning Curves")
plt.legend()
plt.savefig(f"{results_dir}/learning_curves.png")
plt.close()

# -------------------------------
#  8. Evaluation and Confidence Intervals
# -------------------------------
def predict_mc(model, X_tensor, T=50):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(T):
            mean, _ = model(X_tensor.to(device))
            preds.append(mean.cpu().numpy())
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

X_val_tensor = X_tensor[val_ds.indices]
Y_val_tensor = Y_tensor[val_ds.indices]

pred_mean, pred_var = predict_mc(model, X_val_tensor)

# Z-score for % confidence
z = norm.ppf(0.5 + 0.75 / 2)  


pred_std = np.sqrt(pred_var)  # std dev

lower = pred_mean - z * pred_std
upper = pred_mean + z * pred_std

metrics = []

for i, name in enumerate(output_columns):
    y_true = Y_val_tensor[:, i].numpy()
    y_pred = pred_mean[:, i]
    lower_i = lower[:, i]
    upper_i = upper[:, i]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Multiplied by 100
    ci_width = np.mean(upper_i - lower_i)
    
    # Calculate % CI coverage
    coverage = np.mean((y_true >= lower_i) & (y_true <= upper_i)) * 100  # percentage

    metrics.append({
        "Output": name,
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAE": float(mae),
        "MAPE": float(mape),
        "Mean CI Width (85%)": float(ci_width),
        "85% CI Coverage (%)": float(coverage)
    })

    # Plot CI vs True
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True", color="black")
    plt.plot(y_pred, label="Predicted Mean", linestyle="--", color="blue")
    plt.fill_between(np.arange(len(y_pred)), lower_i, upper_i, color='blue', alpha=0.2, label="85% CI")
    plt.title(f"{name} – Prediction with 85% CI")
    plt.xlabel("Sample Index")
    plt.ylabel(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{name}_confidence_plot.png")
    plt.close()

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"{results_dir}/metrics.csv", index=False)

print("✅ Results, plots, and metrics saved.")
