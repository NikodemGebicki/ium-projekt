import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import get_split, save_json
from config import RANDOM_SEED


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()
    print(f"Zbiór treningowy: {len(X_train)} próbek, zbiór testowy: {len(X_test)} próbek")

    pipe = build_pipeline()
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Pipeline: {pipe}")
    print(f"Czas trenowania: {train_time * 1000:.1f} ms")
    print(f"Accuracy scikit-learn na zbiorze testowym: {acc:.4f}")
    save_json("test_sklearn", {"accuracy": acc, "train_time_s": train_time})
