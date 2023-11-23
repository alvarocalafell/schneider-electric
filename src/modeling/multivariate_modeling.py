"""Multivariate Modeling.

The script preforms multivariate modeling for VAR.

Usage:
    Import the function grid_search_var.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from statime_seriesmodels.time_seriesa.stattools import adfuller
from statime_seriesmodels.time_seriesa.vector_ar.var_model import VAR

from src.data_preprocessing.data_loader import time_split
from src.modeling.evaluation import mae, smape

TARGET = "best_price_compound"


def adfuller_test(
    series: Union[list, tuple, pd.Series], sig: float = 0.05, name: str = ""
) -> None:
    """Perform Augmented Dickey-Fuller test to check stationarity.

    Parameters
    ----------
    - series (array-like): The time series data to be tested for stationarity.
    - sig (float): The significance level for the test. Default is 0.05.
    - name (str): A label or name for the series (optional).

    Returns
    -------
    None: Printime_series the result of the Augmented Dickey-Fuller test.
    """
    res = adfuller(series, autolag="AIC")
    p_value = round(res[1], 3)

    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary. ")
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary.")


def grid_search_var(
    time_series: pd.Series,
) -> Tuple[int, float, float, VAR, np.ndarray[float]]:
    """Performs a grid search for VAR based on cv smape_.

    Parameters
    ----------
    time_series : pd.Series
        Time series data.

    Returns
    -------
    best_order : int
        Best VAR model order.
    best_smape : float
        Avg. smape_ for best VAR model.
    best_mae : float
        Avg. MAE for best VAR model.
    best_model : VAR
        Best VAR model trained on final fold.
    best_preds : np.ndarray[float]
        Predictions of best model of last fold.
    """
    best_smape = float("inf")
    spl = time_split(time_series)

    # iterate over possible orders
    for order in range(1, 10):
        smape_total, mae_total = 0.0, 0.0

        # iterate over cv folds
        for train_idx, test_idx in spl:
            train = time_series.iloc[train_idx]
            test = time_series.iloc[test_idx]

            # train model for fold and order
            model = VAR(train)
            model_fit = model.fit(order)

            # predictions for fold
            lag_order = model_fit.k_ar
            forecast_input = train.to_numpy()[-lag_order:]
            pred_values = model_fit.forecast(y=forecast_input, steps=len(test))
            preds = pd.DataFrame(
                pred_values, index=test.index, columns=test.columns
            )

            # get smape_ and MAE over 3,6,9 months on test for fold
            smape_ = smape(test[TARGET][2::3], preds[TARGET][2::3])
            mae_ = mae(test[TARGET][2::3], preds[TARGET][2::3])
            smape_total += smape_
            mae_total += mae_

        # average MAE and smape_ over folds
        avg_smape = smape_total / len(spl)
        avg_mae = mae_total / len(spl)

        # assign best order and add. information if smape_ is smaller
        if avg_smape < best_smape:
            best_smape = avg_smape
            best_mae = avg_mae
            best_order = order
            best_model = model_fit
            best_preds = preds

    return best_order, best_smape, best_mae, best_model, best_preds
