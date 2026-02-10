"""
Genus-stratified counterfactual analysis (partial dependence) of ecosystem resilience
with respect to forest age.

This script loads genus-specific XGBoost model ensembles (trained under repeated
k-fold cross-validation) and computes counterfactual partial dependence profiles
for the Age predictor. For each genus, Age is systematically set to a grid of values
while all other predictors are held at their observed values, and predictions are
averaged across samples within each model.

Predictor variables:
    - SM   : Soil moisture
    - Ta   : Air temperature
    - Rad  : Shortwave radiation
    - Age  : Forest age

Response variable (model output):
    - TAC  : Ecosystem resilience metric
"""

import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


GENUS_NAMES = [
    "Pinus",
    "Quercus",
    "Larix",
    "Cunninghamia",
    "Betula",
    "Picea",
]

PREDICTOR_COLS = ["SM", "Ta", "Rad", "Age"]


def load_training_dataframe(path):
    """
    Load the genus-specific dataset for counterfactual inference.
    """
    return pd.read_parquet(path)


def load_model_ensemble(path):
    """
    Load the genus-specific model ensemble.
    """
    return joblib.load(path)


def compute_counterfactual_pdp(
    models,
    X_df,
    feature_name="Age",
    n_grid=100,
    grid_min=None,
    grid_max=None,
):
    """
    Compute counterfactual partial dependence for age.

    Method
    ------
    For each value in the Age grid:
        1) Set Age to the grid value for all samples.
        2) Predict using each model in the ensemble.
        3) Average predictions across samples, thereby marginalizing over the
           empirical distribution of background climate variables (SM, Ta, Rad).


    Parameters
    ----------
    models : list
        Ensemble of trained models.
    X_df : pandas.DataFrame
        Predictor table containing columns in PREDICTOR_COLS.
    feature_name : str
        Target feature for counterfactual intervention (Age).
    n_grid : int
        Number of grid points for the feature.
    grid_min, grid_max : float or None
        Optional bounds for the grid. If None, bounds are taken from X_df.

    Returns
    -------
    grid : numpy.ndarray
        Feature grid values.
    pdp_by_model : numpy.ndarray
        Per-model partial dependence curves, shape (n_models, n_grid).
    """
    if grid_min is None:
        grid_min = float(X_df[feature_name].min())
    if grid_max is None:
        grid_max = float(X_df[feature_name].max())

    grid = np.linspace(grid_min, grid_max, n_grid)

    n_models = len(models)
    pdp_by_model = np.empty((n_models, n_grid), dtype=float)

    X_cf = X_df.copy()

    for m_idx, model in enumerate(
        tqdm(models, desc=f"Counterfactual response")
    ):
        curve = []
        for v in grid:
            X_cf.loc[:, feature_name] = v
            yhat = model.predict(X_cf)
            curve.append(float(np.mean(yhat)))
        pdp_by_model[m_idx, :] = np.array(curve, dtype=float)

    return grid, pdp_by_model


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Loading paths
    # ------------------------------------------------------------------
    data_dir = "<PATH_TO_TRAINING_DIR>"      # training_{Genus}.parquet
    ensemble_dir = "<PATH_TO_ENSEMBLE_DIR>"  # ensemble_{Genus}.joblib

    pdp_results = {}

    for genus_name in GENUS_NAMES:

        df_path = os.path.join(data_dir, f"training_{genus_name}.parquet")
        df = load_training_dataframe(df_path)

        X_df = df[PREDICTOR_COLS].copy()

        ensemble_path = os.path.join(ensemble_dir, f"ensemble_{genus_name}.joblib")
        models = load_model_ensemble(ensemble_path)

        grid, pdp_by_model = compute_counterfactual_pdp(
            models=models,
            X_df=X_df,
            feature_name="Age",
            n_grid=100,
        )

        pdp_results[genus_name] = {
            "grid": grid,
            "pdp_by_model": pdp_by_model,
        }
