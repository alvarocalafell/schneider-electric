"""Univariate Modeling.

The script preforms univariate modeling for a baseline, recursive methods
such as ARIMA and ETS as well as direct methods such as XGB.

Usage:
    Either run the whole pipeline (see src/main.py) or
    import the function get_best_cv_model.
"""

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

from config.config_modeling import P_RANGE, Q_RANGE, SEASONAL_TERMS, D
from src.data_preprocessing.data_loader import time_split
from src.modeling.evaluation import mae, smape

warnings.filterwarnings("ignore")

# TODO: write part in main.py
# TODO: write function to get final model on all data


def baseline_model(ts: pd.Series) -> Tuple[float, float, np.ndarray[float]]:
    """Create constant prediction for cv folds and get avg. sMAPE.

    Parameters
    ----------
    ts : pd.Series
        Time series data.

    Returns
    -------
    avg_sMAPE : float
        Avg. sMAPE for baseline model.
    avg_MAE : float
        Avg. MAE for baseline model.
    preds : np.ndarray[float]
        Predictions of last fold.
    """
    spl = time_split(ts)
    sMAPE_total, MAE_total = 0.0, 0.0

    for train_idx, test_idx in spl:
        train = ts.iloc[train_idx]
        test = ts.iloc[test_idx]

        preds = np.repeat(train[-1], len(test))
        sMAPE = smape(test[2::3], preds[2::3])
        MAE = mae(test[2::3], preds[2::3])
        sMAPE_total += sMAPE
        MAE_total += MAE

    avg_sMAPE = sMAPE_total / len(spl)
    avg_MAE = MAE_total / len(spl)

    return avg_sMAPE, avg_MAE, preds


def grid_search_arima(
    ts: pd.Series,
    p_values: List[int],
    d: int,
    q_values: List[int],
    seasonal: Tuple[int] = (0, 0, 0, 0),
) -> Tuple[Tuple[int, int, int], float, float, ARIMA]:
    """Performs a grid search for ARIMA based on cv sMAPE.

    Parameters
    ----------
    ts : pd.Series
        Time series data.
    p_values : List[int]
        List of candidate values for p.
    d : int
        d value for difference to make series stationary.
    q_values : List[int]
        List of candidate values for q.
    seasonal : Tuple[int]
        Tuple for seasonal ARIMA.

    Returns
    -------
    best_order : Tuple[int, int, int]
        Best ARIMA model order (p, d, q).
    best_sMAPE : float
        Avg. sMAPE for best ARIMA model.
    best_MAE : float
        Avg. MAE for best ARIMA model.
    best_model : ARIMA
        Best ARIMA model trained on final fold.
    """
    best_sMAPE = float("inf")

    spl = time_split(ts)

    for p in p_values:
        for q in q_values:
            order = (p, d, q)
            sMAPE_total, MAE_total = 0.0, 0.0

            for train_idx, test_idx in spl:
                train = ts.iloc[train_idx]
                test = ts.iloc[test_idx]
                model = ARIMA(train, order=order, seasonal_order=seasonal)
                model_fit = model.fit()
                preds = model_fit.predict(
                    start=test.index[0], end=test.index[-1]
                )
                sMAPE = smape(test[2::3], preds[2::3])
                MAE = mae(test[2::3], preds[2::3])
                sMAPE_total += sMAPE
                MAE_total += MAE

            avg_sMAPE = sMAPE_total / len(spl)
            avg_MAE = MAE_total / len(spl)

            if avg_sMAPE < best_sMAPE:
                best_sMAPE = avg_sMAPE
                best_MAE = avg_MAE
                best_order = order
                best_model = model_fit

    return best_order, best_sMAPE, best_MAE, best_model


def grid_search_ets(
    ts: pd.Series,
    trend: List[str] = ["add", "additive", "multiplicative", None],
    seasonal: List[str] = ["add", "additive", "multiplicative", None],
    seasonal_periods: List[int] = [None, 12, 6, 3],
) -> Tuple[str, str, int, float, float, ExponentialSmoothing]:
    """Performs a grid search for ETS based on cv sMAPE.

    Parameters
    ----------
    ts : pd.Series
        Time series data.
    trend : List[str]
        List of candidate values for trend.
    seasonal : List[str]
        List of candidate values for seasonal component.
    seasonal_periods : List[int]
        List of candidate values for seasonal periods.

    Returns
    -------
    best_trend : str
        Best trend component.
    best_seasonal : str
        Best seasonal component.
    best_seasonal_periods : int
        Best number of seasonal periods.
    best_sMAPE : float
        Avg. sMAPE for best ETS model.
    best_MAE : float
        Avg. MAE for best ETS model.
    best_model : ExponentialSmoothing
        Best ETS model trained on final fold.
    """
    best_sMAPE = float("inf")

    spl = time_split(ts)

    for trend_type in trend:
        for seasonal_type in seasonal:
            for period in seasonal_periods:
                # skip inappropriate param combinations
                if seasonal_type is None and period:
                    continue
                if seasonal_type and period is None:
                    continue

                sMAPE_total, MAE_total = 0.0, 0.0

                for train_idx, test_idx in spl:
                    train = ts.iloc[train_idx]
                    test = ts.iloc[test_idx]

                    ets_model = ExponentialSmoothing(
                        train,
                        trend=trend_type,
                        seasonal=seasonal_type,
                        seasonal_periods=period,
                    )
                    ets_fit = ets_model.fit()

                    preds = ets_fit.predict(
                        start=test.index[0], end=test.index[-1]
                    )
                    sMAPE = smape(test[2::3], preds[2::3])
                    MAE = mae(test[2::3], preds[2::3])

                    sMAPE_total += sMAPE
                    MAE_total += MAE

                avg_sMAPE = sMAPE_total / len(spl)
                avg_MAE = MAE_total / len(spl)

                if avg_sMAPE < best_sMAPE:
                    best_sMAPE = avg_sMAPE
                    best_MAE = avg_MAE
                    best_trend = trend_type
                    best_seasonal = seasonal_type
                    best_seasonal_periods = period
                    best_model = ets_fit

    return (
        best_trend,
        best_seasonal,
        best_seasonal_periods,
        best_sMAPE,
        best_MAE,
        best_model,
    )


def direct_model(
    ts: pd.DataFrame, col: str, horizons: List[int] = [3, 6, 9]
) -> Tuple[float, float, dict[int, XGBRegressor]]:
    """Create direct xgboost models for cv folds and horizons.

    Parameters
    ----------
    ts : pd.DataFrame
        Time series data as df.
    col : str
        Column of this time series.
    horizons : List[int]
        List of months that should be predicted.

    Returns
    -------
    avg_sMAPE : float
        Avg. sMAPE for direct models.
    avg_MAE : float
        Avg. MAE for direct models.
    last_model : dict[int, XGBRegressor]]
        Direct models for each horizon trained on last cv fold.
    """
    spl = time_split(ts)
    sMAPE_total, MAE_total = 0.0, 0.0

    for train_idx, test_idx in spl:
        train = ts.iloc[train_idx]
        test = ts.iloc[np.append(train_idx, test_idx)]

        tests = []
        preds = []

        last_model = {}

        for horizon in horizons:
            X_train = train.copy()
            X_test = test.copy()
            for lag in range(horizon, horizon + 12):
                X_train[f"lag_{lag}"] = X_train[col].shift(lag)
                X_test[f"lag_{lag}"] = X_test[col].shift(lag)

            X_train = X_train.dropna()
            y_train = X_train[col]
            X_train = X_train.drop(columns=col)

            X_test = X_test.dropna()
            y_test = X_test[col]
            X_test = X_test.drop(columns=col)

            model = XGBRegressor(max_depth=3)
            model.fit(X_train, y_train)

            tests.append(y_test[len(y_test) - len(test_idx) + horizon - 1])
            preds.append(
                model.predict(X_test)[
                    len(X_test) - len(test_idx) + horizon - 1
                ]
            )

            last_model[horizon] = model

        sMAPE = smape(np.array(tests), np.array(preds))
        sMAPE_total += sMAPE

        MAE = mae(np.array(tests), np.array(preds))
        MAE_total += MAE

    avg_sMAPE = sMAPE_total / len(spl)
    avg_MAE = MAE_total / len(spl)

    return avg_sMAPE, avg_MAE, last_model


def get_best_cv_model(df: pd.DataFrame) -> dict[str, dict]:
    """Run cv search for best univariate model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of all columns.

    Returns
    -------
    models : dict[str, dict]
        Dict of models with evalutaion information for each column.
    """
    models = {}
    for col in df.columns:
        # create target series
        series = df[col].copy()
        series.index = series.index.to_period("M")
        series = series.dropna()

        print(f"{col}")

        # baseline
        avg_sMAPE, avg_MAE, preds = baseline_model(series)

        models[col] = {
            "baseline": {
                "sMAPE": avg_sMAPE,
                "MAE": avg_MAE,
                "model": preds,
            }
        }

        print("** Baseline **")
        print(f"- Avg. sMAPE (3-6-9 months): {avg_sMAPE:.2f}")

        # arima
        best_order, best_sMAPE, best_MAE, best_model = grid_search_arima(
            series,
            P_RANGE[col],
            D[col],
            Q_RANGE[col],
            SEASONAL_TERMS.get(col, (0, 0, 0, 0)),
        )

        print("** ARIMA **")
        print(
            f"- Best ARIMA Order: {best_order} "
            f"x {SEASONAL_TERMS.get(col, (0, 0, 0, 0))}"
        )
        print(f"- Avg. sMAPE (3-6-9 months): {best_sMAPE:.2f}")

        models[col]["ARIMA"] = {
            "sMAPE": best_sMAPE,
            "MAE": best_MAE,
            "order": best_order,
            "seasonal_order": SEASONAL_TERMS.get(col, (0, 0, 0, 0)),
            "model": best_model,
        }

        # ETS
        (
            best_trend,
            best_seasonal,
            best_seasonal_periods,
            best_sMAPE,
            best_MAE,
            best_model,
        ) = grid_search_ets(series)

        print("** ETS **")
        print(f"- Best Trend: {best_trend}")
        print(f"- Best Seasonal: {best_seasonal}")
        print(f"- Best Seasonal Periods: {best_seasonal_periods}")
        print(f"- Avg. sMAPE (3-6-9 months): {best_sMAPE:.2f}")

        models[col]["ETS"] = {
            "sMAPE": best_sMAPE,
            "MAE": best_MAE,
            "trend": best_trend,
            "seasonal": best_seasonal,
            "seasonal_periods": best_seasonal_periods,
            "model": best_model,
        }

        # direct models
        avg_sMAPE, avg_MAE, model = direct_model(df[[col]].copy(), col)

        print("** XGBOOST **")
        print(f"- Avg. sMAPE (3-6-9 months): {avg_sMAPE:.2f}")

        models[col]["XGB"] = {
            "sMAPE": avg_sMAPE,
            "MAE": avg_MAE,
            "model": model,
        }

        print("---------------")

    # create key selected to point to model with smallest
    # avg. sMAPE across models
    for col in models:
        d = {k: v["sMAPE"] for k, v in models[col].items()}
        models[col]["selected"] = min(d, key=d.get)

    return models
