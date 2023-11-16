from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from config.config_data import DROPNA_COLS, N_FOLDS
from src.data_preprocessing.data_enigneering import handle_na
from src.data_preprocessing.data_loader import load_data, time_split


def data_pipeline(
    path_: Path,
) -> Tuple[
    pd.DataFrame, Union[None, List[Tuple[np.ndarray[int], np.ndarray[int]]]]
]:
    """Performs all data preprocessing steps.

    Parameters
    -------
    path_: Path
            Path to dataframe.

    Returns
    -------
    df : pd.DataFrame
        Data as a dataframe.
    splits : Union[None, List[Tuple[np.ndarray[int], np.ndarray[int]]]]]
                Splits of train and test indices per fold, if any.
    """
    # create initial dataset
    df = load_data(path_)

    # handle missing data
    if DROPNA_COLS:
        df = handle_na(df, DROPNA_COLS)

    # perform time series cv split if wanted
    if N_FOLDS > 1:
        splits = time_split(df, N_FOLDS)
    else:
        splits = None

    return df, splits
