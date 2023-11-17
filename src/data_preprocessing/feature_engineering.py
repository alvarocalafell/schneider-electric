"""Feature Engineering.

The script preforms feature engineering such as grouping of variables by
aggregation, adding time variables, and performing cyclical transformations.

Usage:
    Either run the whole pipeline (see src/main.py) or
    import the function preprocessor.
"""


from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.data_preprocessing.data_loader import load_data


def group_variable(
    df: pd.DataFrame,
    features: List[str],
    new_feature_name: str,
    aggregate_func: str = "mean",
) -> pd.DataFrame:
    """Returns a df with the grouped variables grouped with the grouping func.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe for which we want to compute grouped variable.
    features: List[str]
        List of features to be grouped.
    new_feature_name:
        Name of the new feature
    aggregate_func: str
        Function which we will use to aggregate the variables.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the grouped variables.
    """
    possible_aggs = ["mean", "min", "max"]
    if aggregate_func not in possible_aggs:
        raise ValueError(
            "Invalid aggregate function. Expected one of: %s" % possible_aggs
        )

    if aggregate_func == "mean":
        df[new_feature_name] = df[features].mean(axis=1)
    elif aggregate_func == "min":
        df[new_feature_name] = df[features].min(axis=1)
    elif aggregate_func == "max":
        df[new_feature_name] = df[features].max(axis=1)

    # drop old features
    df = df.drop(features, axis=1)

    return df


def add_time_variables(df: pd.DataFrame, time_var: str) -> pd.DataFrame:
    """Given a time variable, add it to the dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe for which we want to compute the time variable.
    time_var: str
        Time variable we want to add to the dataframe.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the time variable.
    """
    possible_time_vars = ["year", "month", "season", "quarter"]
    if time_var not in possible_time_vars:
        raise ValueError(
            "Invalid time variable. Expected one of: %s" % possible_time_vars
        )

    # create time variable
    if time_var == "year":
        df["year"] = df.index.year
    elif time_var == "month":
        df["month"] = df.index.month
    elif time_var == "season":
        df["season"] = (df.index.month % 12) // 3
    elif time_var == "quarter":
        df["quarter"] = (df.index.month - 1) // 3 + 1

    return df


def cyclical_variable_prep(
    df: pd.DataFrame, cyclical_var: str
) -> pd.DataFrame:
    """Computes sin and cos decomp for cyclical variables.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe for which we want to do sin, cos decomp.
    cyclical_var: str
        Cyclical variable we want to decompose.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the cyclical variable decomposed.
    """
    possible_cyclical_vars = ["month", "season", "quarter"]
    if cyclical_var not in possible_cyclical_vars:
        raise ValueError(
            "Not a cyclical variable. Expected one of: %s"
            % possible_cyclical_vars
        )

    # do cyclical transformation
    if cyclical_var == "month":
        df["month_sin"] = np.sin(df["month"] * 2 * np.pi / 12)
        df["month_cos"] = np.cos(df["month"] * 2 * np.pi / 12)
    elif cyclical_var == "season":
        df["season_sin"] = np.sin(df["season"] * 2 * np.pi / 4)
        df["season_cos"] = np.sin(df["season"] * 2 * np.pi / 4)
    elif cyclical_var == "quarter":
        df["quarter_sin"] = np.sin(df["quarter"] * 2 * np.pi / 4)
        df["quarter_cos"] = np.cos(df["quarter"] * 2 * np.pi / 4)

    # drop old var
    df = df.drop(cyclical_var, axis=1)

    return df


def drop_na_vals(df: pd.DataFrame, drop_variable: str) -> pd.DataFrame:
    """Returns a df with the rows with NaN in the drop_variable column dropped.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe for which we want to drop NaN values.
    drop_variable: str
        Variable which we will check for NaN values.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the dropped values.
    """
    df = df[df[drop_variable].notna()]
    return df


def preprocessor(
    grouping_features: List[List[str]],
    feature_names: List[str],
    aggregate_funcs: List[str],
    time_vars: List[str],
    drop_variables: List[str],
    data_path: Path = None,
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Function which preprocesses the data.

    Parameters
    ----------
    grouping_features: List[List[str]]
        List of groups of features to be grouped.
    feature_names: List[str]
        List of naame of the new grouping features.
    aggregate_func: List[str]
        List of functions we will use to aggregate the groups of features.
    time_vars: List[str]
        Time variables we want to add to the dataframe.
    drop_variables: List[str]
        List of variables to be dropped from the df.
    data_path: Path
        Path to the data.
    df: pd.DataFrame
        Dataframe we want to preprocess.

    Returns
    -------
    df: pd.DataFrame
        Processed dataframe.
    """
    # load data or pass df
    if data_path is None and df is None:
        raise ValueError(
            "Need to give either a dataframe or a path. Please try again."
        )
    if data_path and df:
        raise ValueError(
            "Data Path and df given. Give only one. Please try again."
        )

    if data_path:
        df = load_data(data_path)

    # group features by agg
    for i in range(len(grouping_features)):
        df = group_variable(
            df, grouping_features[i], feature_names[i], aggregate_funcs[i]
        )

    # add time variables
    for var in time_vars:
        df = add_time_variables(df, var)

    # do cyclical transformations
    cyclical_vars = [x for x in time_vars if x != "year"]
    for var in cyclical_vars:
        df = cyclical_variable_prep(df, var)

    # drop columns
    df = df.drop(drop_variables, axis=1)
    return df
