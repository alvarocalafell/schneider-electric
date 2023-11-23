"""Data Loader.

The functions in this script perform data load and a time series split.

Usage:
    Either run the whole pipeline (see src/main.py) or
    import the functions.
"""


from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_data(path_to_file: Path) -> pd.DataFrame:
    """Loads the main data file from csv to a pandas dataframe.

    Parameters
    -------
    path_to_file : Path
                Path to main csv.

    Returns
    -------
    data : pd.DataFrame
            Data as a dataframe.
    """
    data = pd.read_csv(path_to_file)

    # convert time from string to datetime and set it as index
    data.index = pd.to_datetime(data["time"])
    data = data.drop(columns="time")

    return data


def time_split(
    data: pd.DataFrame, n_folds: int = 6, test_size: int = 9
) -> List[Tuple[np.ndarray[int], np.ndarray[int]]]:
    """Creates an extending time series split for data.

    Parameters
    -------
    data : pd.DataFrame
        Data as a dataframe.
    n_folds : int, optional
        Number of time series folds, default is 6.
    test_size : int, optional
        Number of rows in one test test, default is 9.

    Returns
    -------
    all_splits : List[Tuple[np.ndarray[int], np.ndarray[int]]]
                Splits of train and test indices per fold.
    """
    all_splits = []
    split_index = len(data) - n_folds * test_size
    train_ids = np.arange(0, split_index)

    for _ in range(1, n_folds + 1):
        test_ids = np.arange(split_index, split_index + test_size)

        all_splits.append((train_ids, test_ids))
        train_ids = np.append(train_ids, test_ids)

        split_index += test_size

    return all_splits
