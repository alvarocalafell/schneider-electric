import pandas as pd

from config.config_data import DATA_DIR, MAIN_FILE


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / MAIN_FILE)

    return df
