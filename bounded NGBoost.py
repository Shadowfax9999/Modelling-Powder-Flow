import os
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ngboost.distns import Normal
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from ngboost import NGBoost
import shap

# ---------------------------
# Custom Tree Learner
# ---------------------------
class CustomTreeLearner(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree_regressor = DecisionTreeRegressor(max_depth=self.max_depth, random_state=0)

    def fit(self, X, y):
        self.tree_regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.tree_regressor.predict(X)

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth}

# ---------------------------
# Sigmoid-Bounded Distribution
# ---------------------------
class SigmoidBoundedDistribution(Normal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logpdf(self, y, mu, sigma):
        y_sigmoid = expit(y)
        mu_sigmoid = expit(mu)
        return super().logpdf(y_sigmoid, mu_sigmoid, sigma)

# ---------------------------
# Extended NGBoost with loss tracking
# ---------------------------
class NGBoostWithManualLoss(NGBoost):
    def fit_with_loss_tracking(self, X_train, y_train, X_val, y_val):
        self.train_losses = []
        self.val_losses = []

        for i in range(1, self.n_estimators + 1):
            self.n_estimators = i
            super().fit(X_train, y_train)
            pred_train = self.pred_dist(X_train).mean()
            pred_val = self.pred_dist(X_val).mean()

            self.train_losses.append(mean_squared_error(y_train, pred_train))
            self.val_losses.append(mean_squared_error(y_val, pred_val))

        return self

# ---------------------------
# Load and preprocess data
# ---------------------------
data = pd.read_csv("/Users/charliemurray/Documents/all_cohesionless_data/merged_filtered_data.csv")
input_columns = ["gt_c", "carr_index", "dynamic_angle_of_repose", "rpm", "size", 'angle_of_repose', '3rd_order_cubic_term']
output_columns = ["restitution", "sliding_friction", "rolling_friction"]
data = data.dropna(subset=input_columns, how='all')
X_train_full, X_test, y_train_full, y_test = train_test_split(data[input_columns], data[output_columns], test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# ---------------------------
# Custom MAPE
# ---------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true_safe - y_pred) / y_true_safe), axis=0) * 100

# ---------------------------
# Optuna objective
# ---------------------------
def objective(trial, output_name):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.05)
    n_estimators = trial.suggest_int('n_estimators', 100, 400)
    max_depth = trial.suggest_int('max_depth', 2, 3)
    minibatch_frac = trial.suggest_float('minibatch_frac', 0.5, 1.0)
    col_sample = trial.suggest_float('col_sample', 0.5, 0.9)

    custom_learner = CustomTreeLearner(max_depth=max_depth)
    model = NGBoostWithManualLoss(
        Dist=SigmoidBoundedDistribution,
        Base=custom_learner,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        natural_gradient=True,
        minibatch_frac=minibatch_frac,
        col_sample=col_sample,
        verbose=False
    )

    model.fit(X_train, y_train[output_name])
    pred = model.pred_dist(X_val).mean()
    return mean_absolute_error(y_val[output_name], pred)

# ---------------------------
# Run tuning and training
# ---------------------------
study = optuna.create_study(direction='minimize')
best_params = {}

for output_name in output_columns:
    print(f"Tuning hyperparameters for {output_name}...")
    study.optimize(lambda trial: objective(trial, output_name), n_trials=100)
    best_params[output_name] = study.best_params
    print(f"Best hyperparameters for {output_name}: {best_params[output_name]}")
    print(f"Best MAE for {output_name}: {study.best_value}")

# ---------------------------
# Final training and evaluation
# ---------------------------
output_folder = "NGBoost_Model"
os.makedirs(output_folder, exist_ok=True)

for output_name in output_columns:
    print(f"\nTraining final model for {output_name}...")
    params = best_params[output_name]

    custom_learner = CustomTreeLearner(max_depth=params['max_depth'])
    model = NGBoostWithManualLoss(
        Dist=SigmoidBoundedDistribution,
        Base=custom_learner,
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        natural_gradient=True,
        minibatch_frac=params['minibatch_frac'],
        col_sample=params['col_sample'],
        verbose=False
    )

    # Track losses
    model.fit_with_loss_tracking(X_train, y_train[output_name], X_val, y_val[output_name])

    # Final model on full training data
    model.fit(X_train_full, y_train_full[output_name])

    # Predict
    pred_dist = model.pred_dist(X_test)
    pred_mean = pred_dist.mean()
    pred_std = np.sqrt(pred_dist.var)
    y_true = y_test[output_name].values

    # Metrics
    mae = mean_absolute_error(y_true, pred_mean)
    mse = mean_squared_error(y_true, pred_mean)
    r2 = r2_score(y_true, pred_mean)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    # Learning Curves
    plt.figure(figsize=(10, 5))
    plt.plot(model.train_losses, label="Training Loss")
    plt.plot(model.val_losses, label="Validation Loss")
    plt.title(f"Learning Curves - {output_name}")
    plt.xlabel("Boosting Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_learning_curves.png"))
    plt.close()

    # Prediction vs True Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True", color="black")
    plt.plot(pred_mean, label="Predicted", color="blue")
    plt.title(f"Prediction vs True Values - {output_name}")
    plt.xlabel("Sample Index")
    plt.ylabel(output_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_prediction_vs_true.png"))
    plt.close()

    # Confidence Interval Plot (95%)
    lower_bound = pred_mean - 1.96 * pred_std
    upper_bound = pred_mean + 1.96 * pred_std
    x_axis = np.arange(len(pred_mean))

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, y_true, label='True', linewidth=2)
    plt.plot(x_axis, pred_mean, label='Predicted Mean', linewidth=2)
    plt.fill_between(x_axis, lower_bound, upper_bound, color='orange', alpha=0.3, label='95% CI')
    plt.title(f"{output_name.capitalize()} Prediction with 95% Confidence Interval")
    plt.xlabel("Test Sample Index")
    plt.ylabel(output_name.replace("_", " ").capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_confidence_interval.png"))
    plt.close()

    # Prediction Interval (95%) Coverage
    within_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage = np.mean(within_bounds)
    print(f"Prediction Interval (95%) Coverage: {coverage * 100:.2f}%")

    # ---------------------------
    # SHAP Plot Integration
    # ---------------------------
    explainer = shap.Explainer(model.predict, X_train_full)
    shap_values = explainer(X_test)

    # SHAP Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_shap_summary_bar.png"))
    plt.close()

    # SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_shap_beeswarm.png"))
    plt.close()
