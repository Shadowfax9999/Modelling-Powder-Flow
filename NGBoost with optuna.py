import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import optuna
import shap
from ngboost import NGBoost
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# ---------------------------
# Custom Learner Class
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
# Manual Loss Tracking Class
# ---------------------------
class NGBoostWithManualLoss(NGBoost):
    def fit(self, X, Y, X_val=None, Y_val=None, **kwargs):
        self.train_loss = []
        self.val_loss = []
        super().fit(X, Y, **kwargs)
        for i in range(self.n_estimators):
            pred_dist = self.pred_dist(X, i + 1)
            train_loss = -pred_dist.logpdf(Y).mean()
            self.train_loss.append(train_loss)
            if X_val is not None and Y_val is not None:
                pred_val_dist = self.pred_dist(X_val, i + 1)
                val_loss = -pred_val_dist.logpdf(Y_val).mean()
                self.val_loss.append(val_loss)
        return self

# ---------------------------
# Load and split data
# ---------------------------
data = pd.read_csv("/Users/charliemurray/Documents/all_cohesionless_data/merged_filtered_data.csv")
input_columns = ["gt_c", "carr_index", "dynamic_angle_of_repose", "rpm", "size", 'angle_of_repose', '3rd_order_cubic_term']
output_columns = ["restitution", "sliding_friction", "rolling_friction"]

data = data.dropna(subset=input_columns, how='all')
X_train, X_test, y_train, y_test = train_test_split(data[input_columns], data[output_columns], test_size=0.3, random_state=42)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true_safe - y_pred) / y_true_safe), axis=0) * 100

# ---------------------------
# Optuna for hyperparameter tuning
# ---------------------------
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.05)
    n_estimators = trial.suggest_int('n_estimators', 100, 400)
    max_depth = trial.suggest_int('max_depth', 2, 3)
    minibatch_frac = trial.suggest_float('minibatch_frac', 0.5, 1.0)
    col_sample = trial.suggest_float('col_sample', 0.5, 0.9)

    val_mapes = []
    for i in range(y_train.shape[1]):
        custom_learner = CustomTreeLearner(max_depth=max_depth)
        model = NGBoostWithManualLoss(
            Dist=Normal,
            Base=custom_learner,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            natural_gradient=True,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            verbose=False
        )
        model.fit(X_train, y_train.iloc[:, i])
        preds = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test.iloc[:, i], preds)
        val_mapes.append(mape)

    return np.mean(val_mapes)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1)
print("Best hyperparameters:", study.best_params)

# ---------------------------
# Final training and visualization
# ---------------------------
output_folder = "NGBoost_Model"
os.makedirs(output_folder, exist_ok=True)

metrics_df = pd.DataFrame(columns=["Output", "MAE", "MAPE (%)", "RMSE", "R²", "PI Coverage (%)"])
for i, output_name in enumerate(output_columns):
    print(f"\nTraining NGBoost for: {output_name}")

    custom_learner = CustomTreeLearner(max_depth=study.best_params["max_depth"])
    model = NGBoostWithManualLoss(
        Dist=Normal,
        Base=custom_learner,
        n_estimators=study.best_params["n_estimators"],
        learning_rate=study.best_params["learning_rate"],
        natural_gradient=True,
        minibatch_frac=study.best_params["minibatch_frac"],
        col_sample=study.best_params["col_sample"],
        verbose=True
    )

    model.fit(X_train, y_train.iloc[:, i], X_val=X_test, Y_val=y_test.iloc[:, i])

    pred_dist = model.pred_dist(X_test)
    pred_mean = pred_dist.mean()
    pred_std = pred_dist.scale

    lower = pred_mean - 1.64 * pred_std
    upper = pred_mean + 1.64 * pred_std
    x = np.arange(len(pred_mean))

    coverage = np.mean((y_test.iloc[:, i] >= lower) & (y_test.iloc[:, i] <= upper)) * 100

    # Predictive interval plot
    plt.figure(figsize=(12, 5))
    plt.plot(x, y_test.iloc[:, i], label="True", color="black")
    plt.plot(x, pred_mean, label="Predicted", color="blue")
    plt.fill_between(x, lower, upper, color="blue", alpha=0.3, label="90% PI")
    plt.title(f"Predictive Interval - {output_name}")
    plt.xlabel("Sample Index")
    plt.ylabel(output_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_predictive_interval.png"))
    plt.close()

    # Learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(model.train_loss, label="Training Loss", color="blue")
    if model.val_loss:
        plt.plot(model.val_loss, label="Validation Loss", color="red")
    plt.xlabel("Boosting Iteration")
    plt.ylabel("Negative Log Likelihood")
    plt.title(f"Learning Curve - {output_name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_learning_curve.png"))
    plt.close()

    # Metrics
    mae = mean_absolute_error(y_test.iloc[:, i], pred_mean)
    mape = mean_absolute_percentage_error(y_test.iloc[:, i], pred_mean)
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], pred_mean))
    r2 = r2_score(y_test.iloc[:, i], pred_mean)
    metrics_df.loc[i] = [output_name, mae, mape, rmse, r2, coverage]

    # Save model
    joblib.dump(model, os.path.join(output_folder, f"ngboost_{output_name}.pkl"))

    # ---------------------------
    # SHAP Plot Integration (FIXED)
    # ---------------------------
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)

    # Create a new figure and render summary plot into it
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)  # DO NOT show; just build the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_shap_summary_plot.png"))
    plt.close()

metrics_df.to_csv(os.path.join(output_folder, "model_metrics.csv"), index=False)
print("All models, metrics, and plots saved to:", output_folder)
