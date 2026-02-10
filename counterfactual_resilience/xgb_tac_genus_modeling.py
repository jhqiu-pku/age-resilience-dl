"""
Genus-stratified XGBoost modeling of ecosystem resilience as a function of
climate and forest age.

This script applies repeated k-fold cross-validation to train ensembles of
XGBoost regression models separately for individual tree genera.

Predictor variables:
    - SM   : Soil moisture
    - Ta   : Air temperature
    - Rad  : Shortwave radiation
    - Age  : Forest age

Response variable:
    - TAC  : Ecosystem resilience metric
"""

import os
import json
import pandas as pd
from tqdm import tqdm

import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score


# Tree genera included in the analysis
GENUS_NAMES = [
    "Pinus",
    "Quercus",
    "Larix",
    "Cunninghamia",
    "Betula",
    "Picea",
]


def load_training_dataframe(path):
    """
    Load training dataset.
    """
    return pd.read_parquet(path)


def load_params(json_path, genus_name):
    """
    Load XGBoost hyperparameters for a given tree genus.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params[genus_name]


def train_xgb_cv(df, params, n_splits=5, n_repeats=20, random_state=42):
    """
    Train XGBoost regression models using repeated k-fold cross-validation.

    Parameters
    ----------
    df : pandas.DataFrame
        training dataset.
    params : dict
        Hyperparameters used to initialize the XGBoost regressor.
    n_splits : int
        Number of folds in k-fold cross-validation.
    n_repeats : int
        Number of repetitions of the cross-validation procedure.
    random_state : int
        Random seed controlling data partitioning.

    Returns
    -------
    models : list
        Trained XGBoost model ensembles from all cross-validation folds.
    r2_df : pandas.DataFrame
        R² values for training and testing data across folds.
    """
    # Predictor matrix and response vector
    X = df[["SM", "Ta", "Rad", "Age"]]
    y = df["TAC"]

    # Repeated k-fold cross-validation strategy
    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    models = []
    r2_records = []

    total_folds = n_splits * n_repeats
    for train_idx, test_idx in tqdm(
            rkf.split(X, y),
            total=total_folds,
            desc="CV"):

        # Split data into training and testing subsets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Initialize and fit the XGBoost regression model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            # eval_set=[(X_train, y_train)],
            verbose=False,
        )

        # Compute goodness-of-fit metrics
        r2_records.append({
            "r2_train": r2_score(y_train.values.flatten(), model.predict(X_train)),
            "r2_test": r2_score(y_test.values.flatten(), model.predict(X_test)),
        })

        models.append(model)

    r2_df = pd.DataFrame(r2_records)
    return models, r2_df


if __name__ == "__main__":

    # Directory containing genus-specific training datasets
    data_dir = "<PATH_TO_TRAINING_DIR>"
    params_path = os.path.join(data_dir, "xgb_params.json")

    # Train and evaluate models for each tree genus
    for genus_name in GENUS_NAMES:
        df_path = os.path.join(data_dir, f"training_{genus_name}.parquet")
        df = load_training_dataframe(df_path)

        xgb_params = load_params(params_path, genus_name)
        genus_models, genus_r2_df = train_xgb_cv(df, xgb_params)
