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
    data: pd.DataFrame,
    features: List[str],
    new_feature_name: str,
    aggregate_func: str = "mean",
) -> pd.DataFrame:
    """Returns a df with the grouped variables grouped with the grouping func.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe for which we want to compute grouped variable.
    features: List[str]
        List of features to be grouped.
    new_feature_name:
        Name of the new feature
    aggregate_func: str
        Function which we will use to aggregate the variables.

    Returns
    -------
    data: pd.DataFrame
        Dataframe with the grouped variables.
    """
    possible_aggs = ["mean", "min", "max"]
    if aggregate_func not in possible_aggs:
        raise ValueError(
            "Invalid aggregate function. Expected one of: %s" % possible_aggs
        )

    if aggregate_func == "mean":
        data[new_feature_name] = data[features].mean(axis=1)
    elif aggregate_func == "min":
        data[new_feature_name] = data[features].min(axis=1)
    elif aggregate_func == "max":
        data[new_feature_name] = data[features].max(axis=1)

    # drop old features
    data = data.drop(features, axis=1)

    return data


def add_time_variables(data: pd.DataFrame, time_var: str) -> pd.DataFrame:
    """Given a time variable, add it to the dataframe.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe for which we want to compute the time variable.
    time_var: str
        Time variable we want to add to the dataframe.

    Returns
    -------
    data: pd.DataFrame
        Dataframe with the time variable.
    """
    possible_time_vars = ["year", "month", "season", "quarter"]
    if time_var not in possible_time_vars:
        raise ValueError(
            "Invalid time variable. Expected one of: %s" % possible_time_vars
        )

    # create time variable
    if time_var == "year":
        data["year"] = data.index.year
    elif time_var == "month":
        data["month"] = data.index.month
    elif time_var == "season":
        data["season"] = (data.index.month % 12) // 3
    elif time_var == "quarter":
        data["quarter"] = (data.index.month - 1) // 3 + 1

    return data


def cyclical_variable_prep(
    data: pd.DataFrame, cyclical_var: str
) -> pd.DataFrame:
    """Computes sin and cos decomp for cyclical variables.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe for which we want to do sin, cos decomp.
    cyclical_var: str
        Cyclical variable we want to decompose.

    Returns
    -------
    data: pd.DataFrame
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
        data["month_sin"] = np.sin(data["month"] * 2 * np.pi / 12)
        data["month_cos"] = np.cos(data["month"] * 2 * np.pi / 12)
    elif cyclical_var == "season":
        data["season_sin"] = np.sin(data["season"] * 2 * np.pi / 4)
        data["season_cos"] = np.sin(data["season"] * 2 * np.pi / 4)
    elif cyclical_var == "quarter":
        data["quarter_sin"] = np.sin(data["quarter"] * 2 * np.pi / 4)
        data["quarter_cos"] = np.cos(data["quarter"] * 2 * np.pi / 4)

    # drop old var
    data = data.drop(cyclical_var, axis=1)

    return data


def drop_na_vals(data: pd.DataFrame, drop_variable: str) -> pd.DataFrame:
    """Returns df with the rows with NaN in the drop_variable column dropped.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe for which we want to drop NaN values.
    drop_variable: str
        Variable which we will check for NaN values.

    Returns
    -------
    data: pd.DataFrame
        Dataframe with the dropped values.
    """
    data = data[data[drop_variable].notna()]
    return data


def preprocessor(
    grouping_features: List[List[str]],
    feature_names: List[str],
    aggregate_funcs: List[str],
    time_vars: List[str],
    drop_variables: List[str],
    data_path: Path = None,
    data: pd.DataFrame = None,
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
        List of variables to be dropped from the data.
    data_path: Path
        Path to the data.
    data: pd.DataFrame
        Dataframe we want to preprocess.

    Returns
    -------
    data: pd.DataFrame
        Processed dataframe.
    """
    # load data or pass data
    if data_path is None and data is None:
        raise ValueError(
            "Need to give either a dataframe or a path. Please try again."
        )
    if data_path and data:
        raise ValueError(
            "Data Path and data given. Give only one. Please try again."
        )

    if data_path:
        data = load_data(data_path)

    # group features by agg
    for i, gr_feature in enumerate(grouping_features):
        data = group_variable(
            data, gr_feature, feature_names[i], aggregate_funcs[i]
        )

    # add time variables
    for var in time_vars:
        data = add_time_variables(data, var)

    # do cyclical transformations
    cyclical_vars = [x for x in time_vars if x != "year"]
    for var in cyclical_vars:
        data = cyclical_variable_prep(data, var)

    # drop columns
    data = data.drop(drop_variables, axis=1)
    return data
