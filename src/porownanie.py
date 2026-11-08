from pathlib import Path

import numpy as np

from common import load_json
from config import RESULTS_DIR

SRC = Path(__file__).resolve().parent


def count_lines(*files):
    total = 0
    for name in files:
        for line in (SRC / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += 1
    return total


models = {
    "scikit-learn": {
        "cv": load_json("cv_sklearn"),
        "test": load_json("test_sklearn"),
        "lines": count_lines("model_sklearn.py", "cv_sklearn.py"),
    },
    "PyTorch": {
        "cv": load_json("cv_torch"),
        "test": load_json("test_torch"),
        "lines": count_lines("model_torch.py", "cv_torch.py"),
    },
}

header = (
    f"{'Model':<14}| {'Accuracy CV':<17}| {'F1 macro CV':<17}| {'Acc. test':<10}"
    f"| {'Trening [ms]':<13}| {'Czas CV [s]':<12}| {'Linie kodu':<10}"
)
lines = [header, "-" * len(header)]
summary = {}
for name, m in models.items():
    acc = np.array(m["cv"]["accuracy"])
    f1 = np.array(m["cv"]["f1_macro"])
    summary[name] = (acc.mean(), acc.std())
    lines.append(
        f"{name:<14}| {acc.mean():.4f} ± {acc.std():.4f}  | {f1.mean():.4f} ± {f1.std():.4f}  "
        f"| {m['test']['accuracy']:<10.4f}| {m['test']['train_time_s'] * 1000:<13.1f}"
        f"| {m['cv']['cv_time_s']:<12.3f}| {m['lines']:<10}"
    )

(sk_mean, sk_std), (pt_mean, pt_std) = summary["scikit-learn"], summary["PyTorch"]
diff = sk_mean - pt_mean
if abs(diff) <= max(sk_std, pt_std):
    verdict = (
        f"Różnica średniej dokładności wynosi {abs(diff):.4f} i mieści się w odchyleniu "
        f"standardowym, więc oba modele można uznać za porównywalne. Przy podobnej jakości "
        f"prostszy i szybszy w treningu jest pipeline scikit-learn."
    )
elif diff > 0:
    verdict = f"Lepiej wypadł scikit-learn, średnia dokładność wyższa o {diff:.4f}."
else:
    verdict = f"Lepiej wypadł PyTorch, średnia dokładność wyższa o {-diff:.4f}."

text = "\n".join(lines) + "\n\nWniosek: " + verdict + "\n"
print(text)
with open(RESULTS_DIR / "wyniki.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"Zapisano zestawienie do {RESULTS_DIR / 'wyniki.txt'}")
