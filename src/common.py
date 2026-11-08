import json

from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, RESULTS_DIR, TEST_SIZE
from data import load_data


def get_data():
    df = load_data()
    X = df.drop(columns="target").to_numpy(dtype="float32")
    y = df["target"].to_numpy()
    return X, y


def get_split():
    X, y = get_data()
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)


def save_json(name, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(name):
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)
