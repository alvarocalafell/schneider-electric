from pathlib import Path

# data load
DATA_DIR = (
    Path("..") / "hfactory_magic_folders" / "plastic_cost_prediction" / "data"
)
MAIN_FILE = "PA6_cleaned_dataset.csv"

# data preprocessing
DROPNA_COLS = ["best_price_compound"]
N_FOLDS = 10
