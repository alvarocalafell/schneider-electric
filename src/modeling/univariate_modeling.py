import warnings
from typing import List, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from config.config_modeling import P_RANGE, Q_RANGE, SEASONAL_TERMS, D
from src.data_preprocessing.data_pipeline import data_pipeline

warnings.filterwarnings("ignore")


def grid_search_arima(
    ts: pd.Series,
    p_values: List[int],
    d: int,
    q_values: List[int],
    seasonal: Tuple[int] = (0, 0, 0, 0),
) -> Tuple[Tuple[int, int, int], int]:
    """Performs a grid search for ARIMA based on AIC.

    Parameters
    ----------
    ts : pd.Series
        tbd
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
        Best ARIMA model order (p, d, q)
    best_aic : int
        AIC for best ARIMA model order
    """
    best_aic = float("inf")
    best_order = None

    for p in p_values:
        for q in q_values:
            order = (p, d, q)
            model = ARIMA(ts, order=order, seasonal_order=seasonal)
            model_fit = model.fit()
            aic = model_fit.aic

            if aic < best_aic:
                best_aic = aic
                best_order = order

    return best_order, best_aic


def get_arima_model(df: pd.DataFrame) -> dict[str, ARIMA]:
    """Get arima model for each column of df.

    Parameters
    ----------
    df : pd.DataFrame
        tbd

    Returns
    -------
    models : dict[str, ARIMA]
        Dict of univariate models for each column.
    """
    models = {}
    for col in df.columns:
        if col not in Q_RANGE.keys():
            continue

        series = df[col]
        series.index = df.time
        series.index = series.index.to_period("M")

        best_order, best_aic = grid_search_arima(
            series,
            P_RANGE[col],
            D[col],
            Q_RANGE[col],
            SEASONAL_TERMS.get(col, (0, 0, 0, 0)),
        )
        print(f"{col}")
        print(
            f"- Best ARIMA Order: {best_order} "
            f"x {SEASONAL_TERMS.get(col, (0, 0, 0, 0))}"
        )
        print(f"- AIC: {best_aic:.2f}")
        print("---------------")

        model = ARIMA(series, order=best_order)
        model_fit = model.fit()
        models[col] = model_fit

    return models


if __name__ == "__main__":
    df, _ = data_pipeline()
    get_arima_model(df)
