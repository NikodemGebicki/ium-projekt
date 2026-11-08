import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_SEED, ROOT, TEST_SIZE
from data import load_data

df = load_data()
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"Zbiór treningowy: {len(X_train)} próbek, zbiór testowy: {len(X_test)} próbek")

model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)
model.fit(X_train, y_train)
print(f"Wytrenowany model: {model}")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print(f"\nAccuracy na zbiorze testowym: {acc:.4f}\n")
print(report)

results_dir = ROOT / "results" / "lab1"
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "wyniki.txt", "w", encoding="utf-8") as f:
    f.write(f"Model: {model}\n")
    f.write(f"TEST_SIZE={TEST_SIZE}, RANDOM_SEED={RANDOM_SEED}\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

model_path = ROOT / "models" / "decision_tree.joblib"
joblib.dump(model, model_path)
print(f"Zapisano wyniki do {results_dir / 'wyniki.txt'}")
print(f"Zapisano model do {model_path}")
