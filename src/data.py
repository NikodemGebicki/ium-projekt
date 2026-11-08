import pandas as pd
from sklearn.datasets import load_wine

from config import DATA_PATH


def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = load_wine(as_frame=True).frame
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df
