#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 13:02:09 2025

@author: charliemurray
"""
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


    
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
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
# Manual Loss Tracking with Early Stopping
# ---------------------------
class NGBoostWithManualLoss(NGBoost):
    def fit(self, X, Y, X_val=None, Y_val=None, early_stopping_rounds=10, **kwargs):
        self.train_loss = []
        self.val_loss = []
        best_val_loss = float("inf")
        best_iter = 0
        no_improvement_count = 0

        super().fit(X, Y, **kwargs)

        for i in range(self.n_estimators):
            pred_dist = self.pred_dist(X, i + 1)
            train_loss = -pred_dist.logpdf(Y).mean()
            self.train_loss.append(train_loss)

            if X_val is not None and Y_val is not None:
                pred_val_dist = self.pred_dist(X_val, i + 1)
                val_loss = -pred_val_dist.logpdf(Y_val).mean()
                self.val_loss.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iter = i + 1
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= early_stopping_rounds:
                    print(f"Early stopping at iteration {i+1} (best iteration: {best_iter})")
                    break

        # Truncate losses to the best iteration
        self.train_loss = self.train_loss[:best_iter]
        self.val_loss = self.val_loss[:best_iter]
        self.best_iter = best_iter
        return self

# ---------------------------
# Load and split data
# ---------------------------
data = pd.read_csv("/Users/charliemurray/Documents/all_cohesionless_data/merged_filtered_data.csv")
input_columns = ["gt_c", "hausner_ratio", "angle_of_repose", "dynamic_angle_of_repose_(10rpm)", "dynamic_angle_of_repose_(30rpm)", "size", '3rd_order_cubic_term(10rpm)', '3rd_order_cubic_term(30rpm)', "dynamic_angle_of_repose_(50rpm)", '3rd_order_cubic_term(50rpm)']
output_columns = ["restitution", "sliding_friction", "rolling_friction"]

data = data.dropna(subset=input_columns, how='all')
X_train, X_test, y_train, y_test = train_test_split(data[input_columns], data[output_columns], test_size=0.3, random_state=42)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true_safe - y_pred) / y_true_safe), axis=0) * 100

# Define search spaces for each target output
search_spaces = {
    "restitution": {
        "learning_rate": (5e-4, 0.006),
        "n_estimators": (100, 400),       
        "max_depth": (2, 5),
        "minibatch_frac": (0.5, 1.0),
        "col_sample": (0.5, 0.9)
    },
    "sliding_friction": {
        "learning_rate": (5e-4, 0.006),
        "n_estimators": (100, 400),
        "max_depth": (2, 5),
        "minibatch_frac": (0.5, 1.0),
        "col_sample": (0.5, 0.9)
    },
    "rolling_friction": {
        "learning_rate": (1e-3, 0.009),    
        "n_estimators": (500, 1800),              
        "max_depth": (4, 6),                    
        "minibatch_frac": (0.6, 1.0),
        "col_sample": (0.7, 0.9)                  
    }

}



# Store best parameters per output
best_params_per_output = {}

# Loop through each output column
for i, output_name in enumerate(output_columns):
    print(f"\nStarting Optuna tuning for: {output_name}")

    y_train_i = y_train.iloc[:, i]
    y_test_i = y_test.iloc[:, i]
    bounds = search_spaces[output_name]

    def make_objective(y_train_i, y_test_i, bounds):
        def objective(trial):
            learning_rate = trial.suggest_float('learning_rate', *bounds["learning_rate"], log=True)
            n_estimators = trial.suggest_int('n_estimators', *bounds["n_estimators"])
            max_depth = trial.suggest_int('max_depth', *bounds["max_depth"])
            minibatch_frac = trial.suggest_float('minibatch_frac', *bounds["minibatch_frac"])
            col_sample = trial.suggest_float('col_sample', *bounds["col_sample"])

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

            model.fit(X_train, y_train_i, X_val=X_test, Y_val=y_test_i, early_stopping_rounds=15)
            preds = model.predict(X_test)
            return mean_absolute_percentage_error(y_test_i, preds)
        return objective

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(y_train_i, y_test_i, bounds), n_trials=50)

    print(f"Best hyperparameters for {output_name}:", study.best_params)
    best_params_per_output[output_name] = study.best_params

# ---------------------------
# Final training and visualization
# ---------------------------
output_folder = "NGBoost_Model_individual_hyperparameters"
os.makedirs(output_folder, exist_ok=True)
metrics_df = pd.DataFrame(columns=["Output", "MAE", "MAPE (%)", "RMSE", "R²", "PI Coverage (%)"])

for i, output_name in enumerate(output_columns):
    print(f"\nTraining NGBoost for: {output_name}")
    params = best_params_per_output[output_name]

    custom_learner = CustomTreeLearner(max_depth=params["max_depth"])
    model = NGBoostWithManualLoss(
        Dist=Normal,
        Base=custom_learner,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        natural_gradient=True,
        minibatch_frac=params["minibatch_frac"],
        col_sample=params["col_sample"],
        verbose=True
    )

    model.fit(X_train, y_train.iloc[:, i], X_val=X_test, Y_val=y_test.iloc[:, i], early_stopping_rounds=15)

# If early stopping was used, truncate predictions to best_iter
    if hasattr(model, "best_iter"):
        model.n_estimators = model.best_iter


    pred_dist = model.pred_dist(X_test)

    pred_mean = pred_dist.mean()
    pred_std = pred_dist.scale

    lower = pred_mean - 1.64 * pred_std
    upper = pred_mean + 1.64 * pred_std
    x = np.arange(len(pred_mean))
    coverage = np.mean((y_test.iloc[:, i] >= lower) & (y_test.iloc[:, i] <= upper)) * 100

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

    mae = mean_absolute_error(y_test.iloc[:, i], pred_mean)
    mape = mean_absolute_percentage_error(y_test.iloc[:, i], pred_mean)
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], pred_mean))
    r2 = r2_score(y_test.iloc[:, i], pred_mean)
    metrics_df.loc[i] = [output_name, mae, mape, rmse, r2, coverage]

    joblib.dump(model, os.path.join(output_folder, f"ngboost_{output_name}.pkl"))

    plt.figure(figsize=(6, 6))
    plt.errorbar(
        y_test.iloc[:, i],
        pred_mean,
        yerr=1.64 * pred_std,
        fmt='o',
        ecolor='gray',
        elinewidth=1,
        capsize=2,
        alpha=0.5,
        label='Prediction ± 90% CI'
    )
    min_val = min(y_test.iloc[:, i].min(), pred_mean.min())
    max_val = max(y_test.iloc[:, i].max(), pred_mean.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Mean")
    plt.title(f"Parity Plot with Uncertainty - {output_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_parity_plot.png"))
    plt.close()

    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_shap_summary_plot.png"))
    plt.close()


    # Drop rows with NaNs before applying PCA and t-SNE
    X_tsne_pca = X_test.dropna()
    target_values = y_test.iloc[:, i].loc[X_tsne_pca.index]

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tsne_pca)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=1000)
    X_tsne = tsne.fit_transform(X_tsne_pca)

    # Plotting
    norm = plt.Normalize(vmin=target_values.min(), vmax=target_values.max())
    cmap = cm.get_cmap("viridis")

    # PCA plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cmap(norm(target_values)), edgecolor='k', s=40)
    plt.colorbar(label=f"{output_name} value")
    plt.title(f"PCA colored by true {output_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_pca_plot.png"))
    plt.close()

    # t-SNE plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cmap(norm(target_values)), edgecolor='k', s=40)
    plt.colorbar(label=f"{output_name} value")
    plt.title(f"t-SNE colored by true {output_name}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{output_name}_tsne_plot.png"))
    plt.close()


metrics_df.to_csv(os.path.join(output_folder, "model_metrics.csv"), index=False)
print("All models, metrics, and plots saved to:", output_folder)
