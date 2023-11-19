import warnings
from typing import List, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config.config_data import DATA_DIR, MAIN_FILE
from config.config_modeling import P_RANGE, Q_RANGE, SEASONAL_TERMS, D
from src.data_preprocessing.data_loader import load_data, time_split
from src.modeling.evaluation import mae, smape

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
) -> Tuple[str, str, int, float]:
    """Performs a grid search for ETS based on AIC.

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
    best_aic : float
        AIC for the best ETS model.
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


def get_best_cv_model(df: pd.DataFrame) -> dict[str, dict]:
    models = {}
    for col in df.columns:

        series = df[col].copy()
        series.index = series.index.to_period("M")
        series = series.dropna()

        # arima
        best_order, best_sMAPE, best_MAE, best_model = grid_search_arima(
            series,
            P_RANGE[col],
            D[col],
            Q_RANGE[col],
            SEASONAL_TERMS.get(col, (0, 0, 0, 0)),
        )
        print(f"{col}")
        print("** ARIMA **")
        print(
            f"- Best ARIMA Order: {best_order} "
            f"x {SEASONAL_TERMS.get(col, (0, 0, 0, 0))}"
        )
        print(f"- Avg. sMAPE (3-6-9 months): {best_sMAPE:.2f}")
        models[col] = {
            "ARIMA": {
                "sMAPE": best_sMAPE,
                "MAE": best_MAE,
                "order": best_order,
                "seasonal_order": SEASONAL_TERMS.get(col, (0, 0, 0, 0)),
                "model": best_model,
            }
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
        print("---------------")
        models[col]["ETS"] = {
            "sMAPE": best_sMAPE,
            "MAE": best_MAE,
            "trend": best_trend,
            "seasonal": best_seasonal,
            "seasonal_periods": best_seasonal_periods,
            "model": best_model,
        }

        # direct models
        # direct_models = ["elastic net", "xgboost", "random forest"]

    return models


if __name__ == "__main__":
    df = load_data(DATA_DIR / MAIN_FILE)
    get_best_cv_model(df)
