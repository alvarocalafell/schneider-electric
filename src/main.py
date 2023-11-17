"""Main script for pipeline.

The script imports configuration variables from config/. It uses
the functions from the modules of the child folders in src/ to run the
data preprocessing, univariate & multivariate modeling and feature importance.

Usage:
    Run this script to run the whole data pipeline.

Example:
    $ python src/main.py

Note:
    Ensure that the necessary configuration variables are
    properly set in config.config_data.

"""


from config.config_data import (
    DATA_DIR,
    DROP_VARS,
    GROUPING_FUNCS,
    GROUPING_NAMES,
    GROUPING_VARS,
    MAIN_FILE,
    TIME_VARS,
)
from src.data_preprocessing.feature_engineering import preprocessor

if __name__ == "__main__":
    preprocessor(
        data_path=DATA_DIR / MAIN_FILE,
        grouping_features=GROUPING_VARS,
        feature_names=GROUPING_NAMES,
        aggregate_funcs=GROUPING_FUNCS,
        time_vars=TIME_VARS,
        drop_variables=DROP_VARS,
    )
