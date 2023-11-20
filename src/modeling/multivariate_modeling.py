from typing import Union

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

from src.data_preprocessing.data_loader import time_split


def adfuller_test(
    series: Union[list, tuple, pd.Series], sig: float = 0.05, name: str = ""
) -> None:
    """

    Perform Augmented Dickey-Fuller test to check stationarity of a time
    series.

    Parameters:
    - series (array-like): The time series data to be tested for stationarity.
    - sig (float): The significance level for the test. Default is 0.05.
    - name (str): A label or name for the series (optional).

    Returns:
    None: Prints the result of the Augmented Dickey-Fuller test.
    """
    res = adfuller(series, autolag="AIC")
    p_value = round(res[1], 3)

    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary. ")
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary.")


def invert_transformation(
    df_train: pd.DataFrame, df_forecast: pd.DataFrame, third_diff: bool
) -> pd.DataFrame:

    """
    Inverts differencing transformations applied during time series
    forecasting.

    Parameters:
    - df_train (DataFrame): The training DataFrame containing the original
    time series data.
    - df_forecast (DataFrame): The DataFrame containing the forecasted values
    after differencing.
    - third_diff (bool): If True, it assumes that a third-order differencing
    was applied.

    Returns:
    - DataFrame: A DataFrame containing the inverted forecasted values.
    """
    df_fc = df_forecast.copy()
    columns = df_train.columns

    for col in columns:
        if third_diff is True:
            # Roll back 3rd Diff
            df_fc[str(col) + "_2d"] = (
                df_train[col].iloc[-1]
                - df_train[col].iloc[-2]
                + df_fc[str(col) + "_3d"].cumsum()
            )
            df_fc[str(col) + "_1d"] = (
                df_train[col].iloc[-2]
                - df_train[col].iloc[-3]
                + df_fc[str(col) + "_2d"].cumsum()
            )
            df_fc[str(col) + "_forecast"] = (
                df_train[col].iloc[-3] + df_fc[str(col) + "_1d"].cumsum()
            )

        else:
            # Roll back 2nd Diff
            df_fc[str(col) + "_1d"] = (
                df_train[col].iloc[-1] - df_train[col].iloc[-2]
            ) + df_fc[str(col) + "_2d"].cumsum()
            df_fc[str(col) + "_forecast"] = (
                df_train[col].iloc[-1] + df_fc[str(col) + "_1d"].cumsum()
            )
    return df_fc


def grid_search_var(df: pd.DataFrame):

    """
    Find the VAR model with the highest AIC.

    Parameters:
    - df: pandas DataFrame, the input DataFrame with time series data.

    Returns:
    - best_model: statsmodels.tsa.vector_ar.var_model.VARResultsWrapper,
    the best VAR model.
    - best_order: int, the order of the best VAR model.
    """
    best_aic = float("inf")
    best_model = None
    best_order = None

    for order in range(1, 10):
        model = VAR(df)
        results = model.fit(order)
        aic = results.aic
        print(f"Order = {order}, AIC = {aic}")

        if aic < best_aic:
            best_aic = aic
            best_model = results
            best_order = order

    print(f"\nBest Model Order = {best_order}")
    print(f"Best Model AIC = {best_aic}")

    return best_model, best_order

    def get_var_model(
        df: pd.DataFrame, df_stationary: pd.DataFrame, third_diff=False
    ):

        """
        Fit a Vector Autoregressive (VAR) model on a stationary time
        series and generate forecasts.

        Parameters:
        - df (DataFrame): The original time series DataFrame.
        - df_stationary (DataFrame): The DataFrame containing the
        stationary time series data.
        - third_diff (bool): If True, it assumes that a third-order
        differencing was applied.

        Returns:
        - DataFrame: A DataFrame containing the forecasted values.
        """
        df_stationary.dropna(inplace=True)
        spl = time_split(df_stationary)

        for train_idx, test_idx in spl:
            train = df_stationary.iloc[train_idx]
            test = df_stationary.iloc[test_idx]

        best_model, best_order = grid_search_var(train)
        print(best_model.summary())

        lag_order = best_model.k_ar
        forecast_input = df_stationary.values[-lag_order:]

        nobs = len(train)
        fc = best_model.forecast(y=forecast_input, steps=nobs)
        if third_diff is True:
            df_forecast = pd.DataFrame(
                fc, index=df.index[-nobs:], columns=df.columns + "_3d"
            )
        else:
            df_forecast = pd.DataFrame(
                fc, index=df.index[-nobs:], columns=df.columns + "_2d"
            )
        df_results = invert_transformation(train, df_forecast, df)

        fig, axes = plt.subplots(
            nrows=int(len(df.columns) / 2), ncols=2, dpi=150, figsize=(20, 20)
        )
        for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
            df_results[col + "_forecast"].plot(legend=True, ax=ax).autoscale(
                axis="x", tight=True
            )
            test[col][-nobs:].plot(legend=True, ax=ax)
            ax.set_title(col + ": Forecast vs Actuals")
            ax.xaxis.set_ticks_position("none")
            ax.yaxis.set_ticks_position("none")
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)

        plt.tight_layout()
