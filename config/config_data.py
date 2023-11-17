"""Config Data.

This is a config file for the data load, processing and feature engineering.

Usage:
    Change the params here and import them in your script/notebook.
"""


from pathlib import Path

# data load
DATA_DIR = (
    Path("..") / "hfactory_magic_folders" / "plastic_cost_prediction" / "data"
)
MAIN_FILE = "PA6_cleaned_dataset.csv"

# data split
N_FOLDS = 6

# feature engineering
GROUPING_VARS = [
    ["CRUDE_PETRO", "CRUDE_BRENT", "CRUDE_DUBAI", "CRUDE_WTI"],
    [
        "Electricty_Price_France",
        "Electricty_Price_Italy",
        "Electricty_Price_Poland",
        "Electricty_Price_Netherlands",
        "Electricty_Price_Germany",
    ],
    ["NGAS_US", "NGAS_EUR", "NGAS_JP"],
]

GROUPING_NAMES = ["CRUDE_AVG", "Electricity_AVG", "NGAS_AVG"]

GROUPING_FUNCS = ["mean", "mean", "mean"]

TIME_VARS = ["year", "month", "quarter"]

DROP_VARS = [
    "PA6 GLOBAL_ EMEAS _ EUR per TON",
    "Benzene_price",
    "Cyclohexane_price",
    "iNATGAS",
]
