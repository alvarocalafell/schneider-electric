"""Multivariate Modeling.

The script preforms multivariate modeling for VAR.

Usage:
    Import the function grid_search_var.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

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
    None: Prints the result of the Augmented Dickey-Fuller test.
    """
    res = adfuller(series, autolag="AIC")
    p_value = round(res[1], 3)

    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary. ")
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary.")


def grid_search_var(
    ts: pd.Series,
) -> Tuple[int, float, float, VAR, np.ndarray[float]]:
    """Performs a grid search for VAR based on cv sMAPE.

    Parameters
    ----------
    ts : pd.Series
        Time series data.

    Returns
    -------
    best_order : int
        Best VAR model order.
    best_sMAPE : float
        Avg. sMAPE for best VAR model.
    best_MAE : float
        Avg. MAE for best VAR model.
    best_model : VAR
        Best VAR model trained on final fold.
    best_preds : np.ndarray[float]
        Predictions of best model of last fold.
    """
    best_sMAPE = float("inf")
    spl = time_split(ts)

    # iterate over possible orders
    for order in range(1, 10):
        sMAPE_total, MAE_total = 0.0, 0.0

        # iterate over cv folds
        for train_idx, test_idx in spl:
            train = ts.iloc[train_idx]
            test = ts.iloc[test_idx]

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

            # get sMAPE and MAE over 3,6,9 months on test for fold
            sMAPE = smape(test[TARGET][2::3], preds[TARGET][2::3])
            MAE = mae(test[2::3], preds[2::3])
            sMAPE_total += sMAPE
            MAE_total += MAE

        # average MAE and sMAPE over folds
        avg_sMAPE = sMAPE_total / len(spl)
        avg_MAE = MAE_total / len(spl)

        # assign best order and add. information if sMAPE is smaller
        if avg_sMAPE < best_sMAPE:
            best_sMAPE = avg_sMAPE
            best_MAE = avg_MAE
            best_order = order
            best_model = model_fit
            best_preds = preds

    return best_order, best_sMAPE, best_MAE, best_model, best_preds
