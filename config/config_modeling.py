"""Config Modeling.

This is a config file for the modeling process.

Usage:
    Change the params here and import them in your script/notebook.
"""

TARGET = "best_price_compound"

P_RANGE = {
    "best_price_compound": list(range(0, 4)),
    "PA6 GLOBAL_ EMEAS _ EUR per TON": list(range(0, 5)),
    "CRUDE_PETRO": list(range(0, 2)),
    "CRUDE_BRENT": list(range(0, 2)),
    "CRUDE_DUBAI": list(range(0, 2)),
    "CRUDE_WTI": list(range(0, 2)),
    "Benzene_price": list(range(0, 3)),
    "Caprolactam_price": list(range(0, 2)),
    "Cyclohexane_price": list(range(0, 3)),
    "Electricty_Price_Netherlands": list(range(0, 4)),
    "Electricty_Price_France": list(range(0, 4)),
    "Electricty_Price_Italy": list(range(0, 4)),
    "Electricty_Price_Poland": list(range(0, 7)),
    "Electricty_Price_Germany": list(range(0, 5)),
    "NGAS_EUR": list(range(0, 2)),
    "NGAS_US": list(range(0, 6)),
    "NGAS_JP": list(range(0, 9)),
    "iNATGAS": list(range(0, 4)),
    "Inflation_rate_france": list(range(0, 2)),
    "Automotive Value": list(range(0, 5)),
}

Q_RANGE = {
    "best_price_compound": list(range(0, 4)),
    "PA6 GLOBAL_ EMEAS _ EUR per TON": list(range(0, 5)),
    "CRUDE_PETRO": list(range(0, 2)),
    "CRUDE_BRENT": list(range(0, 2)),
    "CRUDE_DUBAI": list(range(0, 2)),
    "CRUDE_WTI": list(range(0, 2)),
    "Benzene_price": list(range(0, 3)),
    "Caprolactam_price": list(range(0, 2)),
    "Cyclohexane_price": list(range(0, 3)),
    "Electricty_Price_Netherlands": list(range(0, 6)),
    "Electricty_Price_France": list(range(0, 6)),
    "Electricty_Price_Italy": list(range(0, 6)),
    "Electricty_Price_Poland": list(range(0, 3)),
    "Electricty_Price_Germany": list(range(0, 3)),
    "NGAS_EUR": list(range(0, 6)),
    "NGAS_US": list(range(0, 6)),
    "NGAS_JP": list(range(0, 6)),
    "iNATGAS": list(range(0, 4)),
    "Inflation_rate_france": list(range(0, 2)),
    "Automotive Value": list(range(0, 2)),
}

D = {
    "best_price_compound": 1,
    "PA6 GLOBAL_ EMEAS _ EUR per TON": 2,
    "CRUDE_PETRO": 1,
    "CRUDE_BRENT": 1,
    "CRUDE_DUBAI": 1,
    "CRUDE_WTI": 1,
    "Benzene_price": 1,
    "Caprolactam_price": 1,
    "Cyclohexane_price": 1,
    "Electricty_Price_Netherlands": 0,
    "Electricty_Price_France": 0,
    "Electricty_Price_Italy": 0,
    "Electricty_Price_Poland": 2,
    "Electricty_Price_Germany": 2,
    "NGAS_EUR": 0,
    "NGAS_US": 1,
    "NGAS_JP": 2,
    "iNATGAS": 0,
    "Inflation_rate_france": 2,
    "Automotive Value": 1,
}

SEASONAL_TERMS = {"Automotive Value": (0, 0, 1, 4)}
