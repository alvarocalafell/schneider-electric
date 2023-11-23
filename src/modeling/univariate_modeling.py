"""Univariate Modeling.

The script preforms univariate modeling for a baseline, recursive methods
such as ARIMA and ETS as well as direct methods such as XGB.

Usage:
    Either run the whole pipeline (see src/main.py) or
    import the function get_best_cv_model.
"""

import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

from config.config_modeling import P_RANGE, Q_RANGE, SEASONAL_TERMS, D
from src.data_preprocessing.data_loader import time_split
from src.modeling.evaluation import mae, smape

warnings.filterwarnings("ignore")


def baseline_model(
    time_series: pd.Series,
) -> Tuple[float, float, np.ndarray[float]]:
    """Create constant prediction for cv folds and get avg. smape_.

    Parameters
    ----------
    time_series : pd.Series
        Time series data.

    Returns
    -------
    avg_smape : float
        Avg. smape_ for baseline model.
    avg_mae : float
        Avg. mae_ for baseline model.
    preds : np.ndarray[float]
        Predictions of last fold.
    """
    spl = time_split(time_series)
    smape_total, mae_total = 0.0, 0.0

    # iterate over cv folds
    for train_idx, test_idx in spl:
        train = time_series.iloc[train_idx]
        test = time_series.iloc[test_idx]

        # predictions are constant of last observation in train data
        preds = np.repeat(train[-1], len(test))

        # get avg. smape_ and mae_ over 3, 6, 9 months on test data for fold
        smape_ = smape(test[2::3], preds[2::3])
        mae_ = mae(test[2::3], preds[2::3])
        smape_total += smape_
        mae_total += mae_

    # average mae_ and smape_ over folds
    avg_smape = smape_total / len(spl)
    avg_mae = mae_total / len(spl)

    return avg_smape, avg_mae, preds


def grid_search_arima(
    time_series: pd.Series,
    p_values: List[int],
    d: int,
    q_values: List[int],
    seasonal: Tuple[int] = (0, 0, 0, 0),
) -> Tuple[Tuple[int, int, int], float, float, ARIMA, np.ndarray[float]]:
    """Performs a grid search for ARIMA based on cv smape_.

    Parameters
    ----------
    time_series : pd.Series
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
    best_smape_ : float
        Avg. smape_ for best ARIMA model.
    best_mae_ : float
        Avg. mae_ for best ARIMA model.
    best_model : ARIMA
        Best ARIMA model trained on final fold.
    best_preds : np.ndarray[float]
        Predictions of best model of last fold.
    """
    best_smape_ = float("inf")
    spl = time_split(time_series)

    # iterate over possible p and q values defining the order
    for p in p_values:
        for q in q_values:
            order = (p, d, q)
            smape_total, mae_total = 0.0, 0.0

            # iterate over cv folds
            for train_idx, test_idx in spl:
                train = time_series.iloc[train_idx]
                test = time_series.iloc[test_idx]

                # train model for fold and order
                model = ARIMA(train, order=order, seasonal_order=seasonal)
                model_fit = model.fit()

                # predictions for fold
                preds = model_fit.predict(
                    start=test.index[0], end=test.index[-1]
                )

                # get smape_ and mae_ over 3,6,9 months on test for fold
                smape_ = smape(test[2::3], preds[2::3])
                mae_ = mae(test[2::3], preds[2::3])
                smape_total += smape_
                mae_total += mae_

            # average mae_ and smape_ over folds
            avg_smape = smape_total / len(spl)
            avg_mae = mae_total / len(spl)

            # assign best order and add. information if smape_ is smaller
            if avg_smape < best_smape_:
                best_smape_ = avg_smape
                best_mae_ = avg_mae
                best_order = order
                best_model = model_fit
                best_preds = preds

    return best_order, best_smape_, best_mae_, best_model, best_preds


def grid_search_etime_series(
    time_series: pd.Series,
    trend: List[str] = ["add", "additive", "multiplicative", None],
    seasonal: List[str] = ["add", "additive", "multiplicative", None],
    seasonal_periods: List[int] = [None, 12, 6, 3],
) -> Tuple[
    str, str, int, float, float, ExponentialSmoothing, np.ndarray[float]
]:
    """Performs a grid search for ETS based on cv smape_.

    Parameters
    ----------
    time_series : pd.Series
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
    best_smape_ : float
        Avg. smape_ for best ETS model.
    best_mae_ : float
        Avg. mae_ for best ETS model.
    best_model : ExponentialSmoothing
        Best ETS model trained on final fold.
    beat_preds : np.ndarray[float]
        Predictions of best model of last fold.
    """
    best_smape_ = float("inf")

    spl = time_split(time_series)

    # iterate over combinations of trend, season and seasonal periods
    for trend_type in trend:
        for seasonal_type in seasonal:
            for period in seasonal_periods:
                # skip inappropriate param combinations
                if seasonal_type is None and period:
                    continue
                if seasonal_type and period is None:
                    continue

                smape_total, mae_total = 0.0, 0.0

                # iterate over cv folds
                for train_idx, test_idx in spl:
                    train = time_series.iloc[train_idx]
                    test = time_series.iloc[test_idx]

                    # train model for fold and param combination
                    etime_series_model = ExponentialSmoothing(
                        train,
                        trend=trend_type,
                        seasonal=seasonal_type,
                        seasonal_periods=period,
                    )
                    etime_series_fit = etime_series_model.fit()

                    # predictions for fold
                    preds = etime_series_fit.predict(
                        start=test.index[0], end=test.index[-1]
                    )

                    # get smape_ and mae_ over 3,6,9 months on test for fold
                    smape_ = smape(test[2::3], preds[2::3])
                    mae_ = mae(test[2::3], preds[2::3])
                    smape_total += smape_
                    mae_total += mae_

                # average mae_ and smape_ over folds
                avg_smape = smape_total / len(spl)
                avg_mae = mae_total / len(spl)

                # assign best model and add. information if smape_ is smaller
                if avg_smape < best_smape_:
                    best_smape_ = avg_smape
                    best_mae_ = avg_mae
                    best_trend = trend_type
                    best_seasonal = seasonal_type
                    best_seasonal_periods = period
                    best_model = etime_series_fit
                    best_preds = preds

    return (
        best_trend,
        best_seasonal,
        best_seasonal_periods,
        best_smape_,
        best_mae_,
        best_model,
        best_preds,
    )


def direct_model(
    time_series: pd.DataFrame, col: str, horizons: List[int] = [3, 6, 9]
) -> Tuple[float, float, dict[int, XGBRegressor], List[float]]:
    """Create direct xgboost models for cv folds and horizons.

    Parameters
    ----------
    time_series : pd.DataFrame
        Time series data as data.
    col : str
        Column of this time series.
    horizons : List[int]
        List of months that should be predicted.

    Returns
    -------
    avg_smape : float
        Avg. smape_ for direct models.
    avg_mae : float
        Avg. mae_ for direct models.
    last_model : dict[int, XGBRegressor]]
        Direct models for each horizon trained on last cv fold.
    preds : List[float]
        Predictions of last fold.
    """
    spl = time_split(time_series)
    smape_total, mae_total = 0.0, 0.0

    # iterate over cv folds
    for train_idx, test_idx in spl:
        train = time_series.iloc[train_idx]
        test = time_series.iloc[np.append(train_idx, test_idx)]

        testime_series = []
        preds = []
        last_model = {}

        # iterate over horizons and create model for each
        for horizon in horizons:
            x_train = train.copy()
            x_test = test.copy()

            # lag horizon+ to create exogenous columns
            for lag in range(horizon, horizon + 12):
                x_train[f"lag_{lag}"] = x_train[col].shift(lag)
                x_test[f"lag_{lag}"] = x_test[col].shift(lag)

            # get train data
            x_train = x_train.dropna()
            y_train = x_train[col]
            x_train = x_train.drop(columns=col)

            # get test data
            x_test = x_test.dropna()
            y_test = x_test[col]
            x_test = x_test.drop(columns=col)

            # fit model for fold and horizon
            model = XGBRegressor(max_depth=3)
            model.fit(x_train, y_train)

            # get target and prediction for horizon
            testime_series.append(
                y_test[len(y_test) - len(test_idx) + horizon - 1]
            )
            preds.append(
                model.predict(x_test)[
                    len(x_test) - len(test_idx) + horizon - 1
                ]
            )

            last_model[horizon] = model

        # get avg. smape_ and mae_ over 3, 6, 9 months on test data for fold
        smape_ = smape(np.array(testime_series), np.array(preds))
        smape_total += smape_
        mae_ = mae(np.array(testime_series), np.array(preds))
        mae_total += mae_

    # average mae_ and smape_ over folds
    avg_smape = smape_total / len(spl)
    avg_mae = mae_total / len(spl)

    return avg_smape, avg_mae, last_model, preds


def get_best_cv_model(data: pd.DataFrame) -> dict[str, dict]:
    """Run cv search for best univariate model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of all columns.

    Returns
    -------
    models : dict[str, dict]
        Dict of models with evalutaion information for each column.
    """
    models = {}
    for col in data.columns:
        # create target series
        series = data[col].copy()
        series.index = series.index.to_period("M")
        series = series.dropna()

        print(f"{col}")

        # baseline
        avg_smape, avg_mae, preds = baseline_model(series)

        models[col] = {
            "baseline": {
                "smape_": avg_smape,
                "mae_": avg_mae,
                "model": "constant",
                "preds": preds,
            }
        }

        print("** Baseline **")
        print(f"- Avg. smape_ (3-6-9 months): {avg_smape:.2f}")

        # arima
        (
            best_order,
            best_smape_,
            best_mae_,
            best_model,
            best_preds,
        ) = grid_search_arima(
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
        print(f"- Avg. smape_ (3-6-9 months): {best_smape_:.2f}")

        models[col]["ARIMA"] = {
            "smape_": best_smape_,
            "mae_": best_mae_,
            "order": best_order,
            "seasonal_order": SEASONAL_TERMS.get(col, (0, 0, 0, 0)),
            "model": best_model,
            "preds": best_preds,
        }

        # ETS
        (
            best_trend,
            best_seasonal,
            best_seasonal_periods,
            best_smape_,
            best_mae_,
            best_model,
            best_preds,
        ) = grid_search_etime_series(series)

        print("** ETS **")
        print(f"- Best Trend: {best_trend}")
        print(f"- Best Seasonal: {best_seasonal}")
        print(f"- Best Seasonal Periods: {best_seasonal_periods}")
        print(f"- Avg. smape_ (3-6-9 months): {best_smape_:.2f}")

        models[col]["ETS"] = {
            "smape_": best_smape_,
            "mae_": best_mae_,
            "trend": best_trend,
            "seasonal": best_seasonal,
            "seasonal_periods": best_seasonal_periods,
            "model": best_model,
            "preds": best_preds,
        }

        # direct models
        avg_smape, avg_mae, model, preds = direct_model(
            data[[col]].copy(), col
        )

        print("** XGBOOST **")
        print(f"- Avg. smape_ (3-6-9 months): {avg_smape:.2f}")

        models[col]["XGB"] = {
            "smape_": avg_smape,
            "mae_": avg_mae,
            "model": model,
            "preds": preds,
        }

        print("---------------")

    # create key selected to point to model with smallest
    # avg. smape_ across models
    for col in models:
        d = {k: v["smape_"] for k, v in models[col].items()}
        models[col]["selected"] = min(d, key=d.get)

    return models


def final_model_univariate(
    data: pd.DataFrame, models: dict[str, dict]
) -> dict[
    str, Union[ARIMA, ExponentialSmoothing, dict[int, XGBRegressor], float]
]:
    """Run cv search for best univariate model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of all columns.
    models : dict[str, dict]
        Dict of models with evalutaion information for each column.

    Returns
    -------
    final_models : dict[str, Union[ARIMA, ExponentialSmoothing,
                        dict[int, XGBRegressor], float]]
        Dict of best model for each column trained on all data
    """
    # get selected models
    models_selected = {
        k: {v["selected"]: v[v["selected"]]} for k, v in models.items()
    }

    # train final model for each column on whole
    final_models = {}
    for col, val in models_selected.items():
        model_type = list(val.keys())[0]

        if model_type == "baseline":
            # define constant value for last point in train data
            final_models[col] = data[col].iloc[-1]

        elif model_type == "ARIMA":
            # fit ARIMA with best order and seasonal order
            model = ARIMA(
                data[col],
                order=val[model_type]["order"],
                seasonal_order=val[model_type]["seasonal_order"],
            )
            model_fit = model.fit()

            final_models[col] = model_fit

        elif model_type == "ETS":
            # fit ETS with best trend, season and seasonal periods
            model = ExponentialSmoothing(
                data[col],
                trend=val[model_type]["trend"],
                seasonal=val[model_type]["seasonal"],
                seasonal_periods=val[model_type]["seasonal_periods"],
            )
            model_fit = model.fit()

            final_models[col] = model_fit

        elif model_type == "XGB":
            # fit direct model as XGB for each horizon
            models_direct = {}
            for horizon in [3, 6, 9]:
                # create lags for each horizon as exogenous columns
                x_train = data[[col]].copy()
                for lag in range(horizon, horizon + 12):
                    x_train[f"lag_{lag}"] = x_train[col].shift(lag)

                # define train data
                x_train = x_train.dropna()
                y_train = x_train[col]
                x_train = x_train.drop(columns=col)

                # fit model for horizon
                model = XGBRegressor(max_depth=3)
                model.fit(x_train, y_train)

                models_direct[horizon] = model

            final_models[col] = models_direct

    return final_models
