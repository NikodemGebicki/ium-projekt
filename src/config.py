import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / os.getenv("DATA_PATH", "data/wine.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_SPLITS = int(os.getenv("N_SPLITS", 5))
EPOCHS = int(os.getenv("EPOCHS", 80))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.01))
RESULTS_DIR = ROOT / "results" / "lab2"
