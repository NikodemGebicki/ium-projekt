import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

from common import get_data, save_json
from config import N_SPLITS, RANDOM_SEED
from model_torch import predict, train_model

X, y = get_data()
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
acc_scores, f1_scores = [], []

print(f"Walidacja krzyżowa PyTorch, KFold k={N_SPLITS}\n")
start = time.perf_counter()
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    model, scaler = train_model(X[train_idx], y[train_idx])
    y_pred = predict(model, scaler, X[test_idx])
    acc = accuracy_score(y[test_idx], y_pred)
    f1 = f1_score(y[test_idx], y_pred, average="macro")
    acc_scores.append(acc)
    f1_scores.append(f1)
    print(f"Fold {fold}:  accuracy = {acc:.4f}   F1 macro = {f1:.4f}")
cv_time = time.perf_counter() - start

acc_scores = np.array(acc_scores)
f1_scores = np.array(f1_scores)
print(f"\nAccuracy: średnia = {acc_scores.mean():.4f}, odchylenie = {acc_scores.std():.4f}")
print(f"F1 macro: średnia = {f1_scores.mean():.4f}, odchylenie = {f1_scores.std():.4f}")
print(f"Czas walidacji krzyżowej: {cv_time:.3f} s")

save_json("cv_torch", {
    "accuracy": acc_scores.tolist(),
    "f1_macro": f1_scores.tolist(),
    "cv_time_s": cv_time,
})
