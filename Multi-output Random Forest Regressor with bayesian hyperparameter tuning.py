import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import joblib
import optuna
from xgboost import XGBRegressor

# Load dataset
data = pd.read_csv("/Users/charliemurray/Documents/all_cohesionless_data/merged_data.csv")

# Extract inputs and outputs
input_columns = ["gt_c", "hausner_ratio", "dynamic_angle_of_repose", "rpm", "size", 'angle_of_repose']  
output_columns = ["restitution", "sliding_friction", "rolling_friction"]  

# Ensure at least one input is present per row
data = data.dropna(subset=input_columns, how='all')  # Keeps rows where at least one input is available

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data[input_columns], data[output_columns], test_size=0.3, random_state=42
)

# Objective function for Optuna (Bayesian optimization)
def objective(trial):
    # Define the hyperparameters you want to optimize
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 0.1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600)
    }

    # Create the XGBRegressor model
    model = XGBRegressor(
        **params, 
        random_state=42, 
        missing=np.nan
    )

    # Train the model (Fixed fit method)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(np.nan_to_num(y_test), np.nan_to_num(y_pred), multioutput='raw_values')
    mse = mean_squared_error(np.nan_to_num(y_test), np.nan_to_num(y_pred), multioutput='raw_values')
    rmse = np.sqrt(mse)
    
    return np.mean(rmse)  # Optimize for RMSE

# Create a study for Bayesian optimization using Optuna
study = optuna.create_study(direction='minimize')  # Minimize RMSE
study.optimize(objective, n_trials=150)  # Number of trials to run the optimization

# Print the best hyperparameters found
print(f"Best hyperparameters: {study.best_params}")

# Best model from the study
best_model = XGBRegressor(**study.best_params, random_state=42, missing=np.nan)

# Train the best model with Learning Curve Tracking
best_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],  # Validation tracking
    verbose=False
)

# Extract training history
results = best_model.evals_result()

# Plot Learning Curve
plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['rmse'], label="Training RMSE", color='blue')
plt.plot(results['validation_1']['rmse'], label="Validation RMSE", color='red')
plt.xlabel("Boosting Rounds")
plt.ylabel("RMSE")
plt.title("XGBoost Learning Curve")
plt.legend()
plt.grid()

# Save Learning Curve
results_dir = "results"
models_dir = "models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, f"learning_curve_{timestamp}.png"))
plt.show()

# Predictions with the best model
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(np.nan_to_num(y_test), np.nan_to_num(y_pred), multioutput='raw_values')
mse = mean_squared_error(np.nan_to_num(y_test), np.nan_to_num(y_pred), multioutput='raw_values')
rmse = np.sqrt(mse)
r2 = r2_score(np.nan_to_num(y_test), np.nan_to_num(y_pred), multioutput='raw_values')

# Compute MAPE correctly
y_test_safe = y_test.replace(0, np.nan)  # Avoid division by zero
mape = np.nanmean(np.abs((y_test_safe - y_pred) / y_test_safe), axis=0) * 100  # Convert to %

# Save metrics
metrics_file = os.path.join(results_dir, f"model_metrics_{timestamp}.csv")
metrics_df = pd.DataFrame({
    "Output Parameter": output_columns,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE (%)": mape,
    "R² Score": r2
})
metrics_df.to_csv(metrics_file, index=False)
print(f"Model metrics saved to {metrics_file}")

# Visualization of evaluation metrics
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.bar(output_columns, mae, color='blue', alpha=0.7)
plt.ylabel("MAE")
plt.title("Mean Absolute Error (MAE) per Output Parameter")
plt.xticks(rotation=45)

plt.subplot(1, 4, 2)
plt.bar(output_columns, mape, color='orange', alpha=0.7)
plt.ylabel("MAPE (%)")
plt.title("Mean Absolute Percentage Error (MAPE) per Output Parameter")
plt.xticks(rotation=45)

plt.subplot(1, 4, 3)
plt.bar(output_columns, r2, color='green', alpha=0.7)
plt.ylabel("R² Score")
plt.title("R² Score per Output Parameter")
plt.xticks(rotation=45)

plt.subplot(1, 4, 4)
plt.bar(output_columns, rmse, color='red', alpha=0.7)
plt.ylabel("RMSE")
plt.title("Root Mean Squared Error (RMSE) per Output Parameter")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"evaluation_metrics_{timestamp}.png"))
plt.show()

# Save trained model
model_file = os.path.join(models_dir, f"trained_model_{timestamp}.pkl")
joblib.dump(best_model, model_file)
print(f"Trained model saved to {model_file}")

# Feature importance analysis using SHAP with a background dataset
background = X_train.sample(n=min(100, len(X_train)), random_state=42)  # Use at most 100 samples
explainer = shap.Explainer(best_model, background, feature_perturbation="interventional")
shap_values = explainer(X_test)

# Plot SHAP summary plot
plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, X_test, feature_names=input_columns, show=False)
plt.title("Feature Importance Analysis (SHAP)")
plt.savefig(os.path.join(results_dir, f"shap_summary_{timestamp}.png"))
plt.show()
