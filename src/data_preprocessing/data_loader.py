from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def load_data(path_to_file: Path) -> pd.DataFrame:
    """Loads the main data file from csv to a pandas dataframe.

    Parameters
    -------
    None : None

    Returns
    -------
    df : pd.DataFrame
            Data as a dataframe.
    """
    df = pd.read_csv(path_to_file)

    # convert time from string to datetime
    df["time"] = pd.to_datetime(df["time"])

    return df


def time_split(
    df: pd.DataFrame, n_folds: int = 5
) -> List[Tuple[np.ndarray[int], np.ndarray[int]]]:
    """Creates time series split for data.

    Parameters
    -------
    df : pd.DataFrame
        Data as a dataframe.
    n_folds : int, optional
        Number of time series folds, default is 5.

    Returns
    -------
    all_splits : List[Tuple[np.ndarray[int], np.ndarray[int]]]
                Splits of train and test indices per fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)
    all_splits = list(tscv.split(df))
    return all_splits
