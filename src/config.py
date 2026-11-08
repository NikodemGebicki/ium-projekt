from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "wine.csv"
TEST_SIZE = 0.2
RANDOM_SEED = 42
