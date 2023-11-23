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

import pandas as pd
from dateutil.relativedelta import relativedelta

from config.config_data import DATA_DIR, MAIN_FILE
from config.config_modeling import TARGET
from src.data_preprocessing.data_loader import load_data
from src.modeling.univariate_modeling import (
    final_model_univariate,
    get_best_cv_model,
)


def main() -> None:
    """Runs modeling pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # univariate modeling
    df = load_data(DATA_DIR / MAIN_FILE)
    models = get_best_cv_model(df)
    univariate_models = final_model_univariate(df, models)

    # get univariate model for target variable
    univariate_model_target = univariate_models[TARGET]

    # predict for 3,6,9 months into the future
    start_idx = df.index[-1] + relativedelta(months=1)
    end_idx = df.index[-1] + relativedelta(months=9)
    preds = univariate_model_target.predict(start=start_idx, end=end_idx)[2::3]

    # save predictions as csv
    out_csv = pd.DataFrame(
        {
            "time": [
                df.index[-1] + relativedelta(months=3),
                df.index[-1] + relativedelta(months=6),
                end_idx,
            ],
            "best_price_compound": preds,
        }
    )
    out_csv.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
