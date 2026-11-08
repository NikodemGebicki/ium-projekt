import shutil

import pandas as pd

from data import load_data

pd.set_option("display.width", shutil.get_terminal_size().columns)
pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 2)

df = load_data()

print("Pierwsze wiersze zbioru:")
print(df.head())
print()
print("Rozmiar zbioru:", df.shape)
print("Liczność klas:", df["target"].value_counts().sort_index().to_dict())
print()
print("Statystyki opisowe:")
print(df.describe().T)
