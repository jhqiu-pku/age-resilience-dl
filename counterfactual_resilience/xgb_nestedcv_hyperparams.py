"""
Nested cross-validation with randomized hyperparameter search for XGBoost.

1) An outer repeated K-fold cross-validation to generate multiple
   independent training partitions.
2) An inner cv within each outer split
   to select optimal hyperparameters via random sampling from a discrete
   candidate space, with explicit de-duplication of sampled configurations.

Inputs
------
- X : feature matrix (pandas.DataFrame)
- y : target vector (pandas.Series)

Outputs
-------
- best_params_df : the selected hyperparameter set
"""


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
import xgboost as xgb


def randomized_search_cv(
    estimator,
    param_distributions,
    X,
    y,
    n_iter=64,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    random_state=42,
):
    """
    Perform randomized hyperparameter search with inner cv.

    Unique parameter combinations are randomly sampled from a discrete
    candidate space and evaluated using K-fold cross-validation.
    """
    # Random number generator for parameter sampling
    rng = np.random.RandomState(random_state)

    # Inner CV used to estimate model performance for each parameter set
    inner_cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Flatten the search space for sampling
    keys = list(param_distributions.keys())
    space = [param_distributions[k] for k in keys]

    # Track tried configurations to avoid duplicate evaluations
    tried_param_set = set()

    # Keep the best-performing configuration under inner CV
    best_score = -np.inf
    best_params = None

    n_success = 0
    while n_success < n_iter:
        # Randomly sample one hyperparameter configuration
        params = {k: rng.choice(v) for k, v in zip(keys, space)}
        param_key = tuple(params[k] for k in keys)

        # Skip configurations that have already been evaluated
        if param_key in tried_param_set:
            continue
        tried_param_set.add(param_key)

        # Clone the base estimator to ensure independence between trials
        est = clone(estimator)
        est.set_params(**params)

        # Evaluate the configuration using inner cross-validation
        scores = cross_val_score(
            est,
            X=X,
            y=y,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
        )

        mean_score = float(np.mean(scores))

        # Update the best configuration if performance improves
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        n_success += 1

    if best_params is None:
        raise RuntimeError("Randomized search failed to evaluate any configuration.")

    return best_params


def run_nested_cv_xgb(
    X,
    y,
    candidate_params,
    outer_splits=5,
    outer_repeats=20,
    inner_splits=5,
    n_iter=64,
    scoring="r2",
    random_state=42,
    search_n_jobs=-1,
):
    """
    Run nested cross-validation for XGBoost and return hyperparameters
    selected in each outer fold.

    The outer loop provides an unbiased data partitioning, while the inner
    loop identifies optimal hyperparameters based solely on training data.
    """
    best_params_per_fold = []

    # Outer repeated K-fold to stabilize hyperparameter selection
    outer_cv = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_state,
    )

    for fold_idx, (train_idx, _) in enumerate(outer_cv.split(X, y)):
        # Split data for the current outer fold
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # Base estimator used for hyperparameter evaluation
        # Parallelism is controlled at the CV level
        base_est = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state + fold_idx,
            n_jobs=1,
        )

        # Inner randomized search to select hyperparameters
        best_params = randomized_search_cv(
            estimator=base_est,
            param_distributions=candidate_params,
            X=X_train,
            y=y_train,
            n_iter=n_iter,
            scoring=scoring,
            cv=inner_splits,
            n_jobs=search_n_jobs,
            random_state=random_state + fold_idx,
        )

        best_params_per_fold.append(best_params)

    # selected hyperparameters
    return pd.DataFrame(best_params_per_fold)


if __name__ == "__main__":
    # data-loading
    X = pd.read_parquet("<PATH_TO_FEATURE_TABLE>")
    y = pd.read_parquet("<PATH_TO_TARGET_TABLE>")["TAC"]

    xgb_candidate_params = {
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "n_estimators": [100, 200, 300, 500, 1000],
        "subsample": [0.5, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.01, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0.01, 0.1, 1, 10],
    }

    best_params_df = run_nested_cv_xgb(
        X=X,
        y=y,
        candidate_params=xgb_candidate_params,
        outer_splits=5,
        outer_repeats=20,
        inner_splits=5,
        n_iter=64,
        scoring="r2",
        random_state=42,
        search_n_jobs=-1,
    )
