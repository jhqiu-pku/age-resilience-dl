## Climate change-induced resilience loss mitigated by young forest age in China

This repository contains the supporting code for the paper:

> Qiu *et al.* (2026), **Climate change-induced resilience loss mitigated by young forest age in China**, (under review)

### Overview

This study evaluates ecosystem resilience dynamics across China (1983–2023) using the lag-1 temporal autocorrelation (TAC) of gross primary productivity (GPP). A data-driven GPP product is generated using a transfer-learning framework, and ecosystem resilience change is then attributed to long-term climate change and forest age dynamics.

The repository is structured as follows:
```
|- transfer_learning_gpp/          # transfer learning model structure
|  |- corecode/
|  |- ealstm.py
|  |- model_structure.py
|  |- multiLoss.py
|- resilience_estimation/          # Ecosystem resilience estimation
|  |- get_tac_moving_windows.py
|  |- get_tac_binned.py
|  |- trend_analysis.py
|- counterfactual_resilience/      # Attribution and counterfactual analyses
|  |- xgb_tac_genus_modeling.py
|  |- xgb_nestedcv_hyperparams.py
|  |- xgb_counterfactual_analysis.py
```

###  System requirements

- operating systems: windows 11
- software version: Python 3.11.5

###  Installation guide

Install dependencies:

`pip install -r requirements.txt`

### Instructions to run on data

- Prepare monthly GPP inputs, then estimate TAC in 5-year moving windows using `resilience_estimation/get_tac_moving_windows.py`.
- Compute pixel-wise trends (e.g., δTAC and climate trends) using `resilience_estimation/trend_analysis.py`.
- Train genus-specific XGBoost ensembles (predictors: SM, Ta, Rad, Age; target: TAC) using `counterfactual_resilience/xgb_tac_genus_modeling.py`.
- Derive age counterfactual response curves (partial dependence) using `counterfactual_resilience/xgb_counterfactual_analysis.py`.

### Expected output

- Moving-window TAC arrays and associated significance estimates
- Temporal trends (δTAC and climate-variable trends)
- Genus-specific XGBoost model ensembles
- Counterfactual age–TAC response curves for attribution analyses

