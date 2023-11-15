from pathlib import Path

DATA_PATH = Path("../../data") / "PA6_cleaned_dataset.csv"

OUT_PATH = Path("../../data") / "PA6_processed_dataset.csv"

TO_DATETIME = True

SEASON_DICT = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
    12: "Winter",
}

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

TIME_VARS = ["year"]

DROP_NA_VARS = ["best_price_compound"]

DROP_VARS = [
    "PA6 GLOBAL_ EMEAS _ EUR per TON",
    "Benzene_price",
    "Cyclohexane_price",
    "iNATGAS",
]
