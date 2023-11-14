from typing import List

import pandas as pd


def handle_na(df: pd.DataFrame, dropna_cols: List[str]) -> pd.DataFrame:
    """Methods to handle missing values, i.e. dropping rows based on subset.

    Parameters
    -------
    df : pd.DataFrame
            Data as a dataframe.
    dropna_cols :  List[str]
            List of columns that are used to drop missing values.

    Returns
    -------
    df : pd.DataFrame
            Transformed data where methods to na values where applied.
    """
    df = df.dropna(subset=dropna_cols)
    return df
