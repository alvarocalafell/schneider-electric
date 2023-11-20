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
from config.config_modeling import TARGET
from src.data_preprocessing.data_loader import load_data
from src.data_preprocessing.feature_engineering import preprocessor
from src.modeling.univariate_modeling import (
    final_model_univariate,
    get_best_cv_model,
)

if __name__ == "__main__":
    # univariate modeling
    df = load_data(DATA_DIR / MAIN_FILE)
    models = get_best_cv_model(df)
    univariate_models = final_model_univariate(df, models)

    # get univariate model for target variable
    univariate_model_target = {TARGET: univariate_models[TARGET]}

    # get univariate models for exogenous columns
    univariate_models.pop(TARGET)

    # feature engineering
    preprocessor(
        data_path=DATA_DIR / MAIN_FILE,
        grouping_features=GROUPING_VARS,
        feature_names=GROUPING_NAMES,
        aggregate_funcs=GROUPING_FUNCS,
        time_vars=TIME_VARS,
        drop_variables=DROP_VARS,
    )
