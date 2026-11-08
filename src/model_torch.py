import time

import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from common import get_split, save_json
from config import EPOCHS, LEARNING_RATE, RANDOM_SEED


class WineNet(nn.Module):
    def __init__(self, n_features, n_classes, hidden1=32, hidden2=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(X_train, y_train, n_classes=3, epochs=EPOCHS, lr=LEARNING_RATE, verbose=False):
    torch.manual_seed(RANDOM_SEED)
    scaler = StandardScaler().fit(X_train)
    X_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    model = WineNet(X_t.shape[1], n_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"Epoka {epoch:3d}/{epochs}   loss = {loss.item():.4f}")
    return model, scaler


def predict(model, scaler, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(scaler.transform(X), dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()
    print(f"Zbiór treningowy: {len(X_train)} próbek, zbiór testowy: {len(X_test)} próbek")
    print(f"PyTorch {torch.__version__}, epoki: {EPOCHS}, learning rate: {LEARNING_RATE}\n")

    start = time.perf_counter()
    model, scaler = train_model(X_train, y_train, verbose=True)
    train_time = time.perf_counter() - start

    y_pred = predict(model, scaler, X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{model}")
    print(f"Czas trenowania: {train_time * 1000:.1f} ms")
    print(f"Accuracy PyTorch na zbiorze testowym: {acc:.4f}")
    save_json("test_torch", {"accuracy": acc, "train_time_s": train_time})
