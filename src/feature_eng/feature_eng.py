from pathlib import Path
from typing import List

import pandas as pd

from config.config_data_preprocessing import (
    DATA_PATH,
    DROP_NA_VARS,
    DROP_VARS,
    GROUPING_FUNCS,
    GROUPING_NAMES,
    GROUPING_VARS,
    OUT_PATH,
    SEASON_DICT,
    TIME_VARS,
    TO_DATETIME,
)


def load_data(data_path: Path) -> pd.DataFrame:
    """Given a path, loads the data.

    Parameters
    ----------
    data_path: Path
        Path to the data.

    Returns
    -------
    df: pd.DataFrame
        Dataframe obtained from the data.
    """
    df = pd.read_csv(data_path)
    return df


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
    if aggregate_func == "min":
        df[new_feature_name] = df[features].min(axis=1)
    if aggregate_func == "max":
        df[new_feature_name] = df[features].max(axis=1)
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
    possible_time_vars = ["year", "month", "season"]
    if time_var not in possible_time_vars:
        raise ValueError(
            "Invalid time variable. Expected one of: %s" % possible_time_vars
        )
    if time_var == "season" and "month" not in list(df.columns):
        raise ValueError(
            "Impossible to compute season without having month column. \
                Please create month column first."
        )
    if time_var == "year":
        df["year"] = [df["time"][i].year for i in range(len(df))]
    if time_var == "month":
        df["month"] = [df["time"][i].month for i in range(len(df))]
    if time_var == "season":
        df["Season"] = df["month"].apply(lambda x: SEASON_DICT[x])
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
    data_path: Path,
    to_datetime: bool,
    grouping_features: List[List[str]],
    feature_names: List[str],
    aggregate_funcs: List[str],
    time_vars: List[str],
    drop_na_variables: List[str],
    drop_variables: List[str],
) -> pd.DataFrame:
    """Function which preprocesses the data.

    Parameters
    ----------
    data_path: Path
        Path to the data.
    to_datetime: bool
        Specifies whether the time column should be turned to datetime.
    grouping_features: List[List[str]]
        List of groups of features to be grouped.
    feature_names: List[str]
        List of naame of the new grouping features.
    aggregate_func: List[str]
        List of functions we will use to aggregate the groups of features.
    time_vars: List[str]
        Time variables we want to add to the dataframe.
    drop_na_variables: List[str]
        Variables which we will check for NaN values.
    drop_variables: List[str]
        List of variables to be dropped from the df.

    Returns
    -------
    df: pd.DataFrame
        Processed dataframe.
    """
    df = load_data(data_path)
    if to_datetime:
        df["time"] = pd.to_datetime(df["time"])
    i = 0
    for i in range(len(grouping_features)):
        df = group_variable(
            df, grouping_features[i], feature_names[i], aggregate_funcs[i]
        )
        i = i + 1
    for var in time_vars:
        df = add_time_variables(df, var)
    for var in drop_na_variables:
        df = drop_na_vals(df, var)
    df = df.drop(drop_variables, axis=1)
    return df


if __name__ == "__main__":
    data = preprocessor(
        DATA_PATH,
        TO_DATETIME,
        GROUPING_VARS,
        GROUPING_NAMES,
        GROUPING_FUNCS,
        TIME_VARS,
        DROP_NA_VARS,
        DROP_VARS,
    )
    data.to_csv(OUT_PATH, index=False)
