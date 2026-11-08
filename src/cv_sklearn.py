import time

from sklearn.model_selection import KFold, cross_val_score

from common import get_data, save_json
from config import N_SPLITS, RANDOM_SEED
from model_sklearn import build_pipeline

X, y = get_data()
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

start = time.perf_counter()
acc_scores = cross_val_score(build_pipeline(), X, y, cv=kf, scoring="accuracy")
cv_time = time.perf_counter() - start
f1_scores = cross_val_score(build_pipeline(), X, y, cv=kf, scoring="f1_macro")

print(f"Walidacja krzyżowa scikit-learn, KFold k={N_SPLITS}\n")
for fold, (acc, f1) in enumerate(zip(acc_scores, f1_scores), start=1):
    print(f"Fold {fold}:  accuracy = {acc:.4f}   F1 macro = {f1:.4f}")

print(f"\nAccuracy: średnia = {acc_scores.mean():.4f}, odchylenie = {acc_scores.std():.4f}")
print(f"F1 macro: średnia = {f1_scores.mean():.4f}, odchylenie = {f1_scores.std():.4f}")
print(f"Czas walidacji krzyżowej: {cv_time:.3f} s")

save_json("cv_sklearn", {
    "accuracy": acc_scores.tolist(),
    "f1_macro": f1_scores.tolist(),
    "cv_time_s": cv_time,
})
