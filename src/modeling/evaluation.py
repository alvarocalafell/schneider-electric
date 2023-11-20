from typing import Union

import numpy as np
import pandas as pd


def smape(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """Calculates sMAPE between true and predicted values.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Array of true values.
    y_pred : Union[pd.Series, np.ndarray]
        Array of predicted values.

    Returns
    -------
    float
        SMAPE value, a percentage measure of the accuracy of the prediction.
    """
    return (
        100
        * np.sum(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
        )
        / len(y_true)
    )


def mae(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """Calculates MAE between true and predicted values.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Array of true values.
    y_pred : Union[pd.Series, np.ndarray]
        Array of predicted values.

    Returns
    -------
    float
        Mean Absolute Error between y_true and y_pred.
    """
    return np.mean(np.abs(y_pred - y_true))
