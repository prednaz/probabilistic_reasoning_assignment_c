from common import to_identifier_suffix
from math import sqrt
import numpy as np
import pandas as pd

# `model.csv` contains the ouput of `az.summary` with "p" prepended
model = pd.read_csv("model.csv", sep='\s+', index_col=0)[["mean", "sd"]]

def predict(row):
    row = row[1]
    species = to_identifier_suffix(row["species"])
    return model.loc[f"a{species}", "mean"] + model.loc[f"b{species}", "mean"] * (row["d18_O"] - row["d18_O_w"])

def standard_deviation(row):
    row = row[1]
    species = to_identifier_suffix(row["species"])
    return sqrt(
        model.loc["sigma", "mean"]**2 +
        model.loc[f"a{species}", "sd"]**2 +
        product_variance(
            model.loc[f"b{species}", "mean"],
            (row["d18_O"] - row["d18_O_w"]),
            model.loc[f"b{species}", "sd"],
            row["d18_O_sd"]**2 + row["d18_O_w_sd"]**2
        )
    )

def product_variance(a, b, a_variance, b_variance):
    return a_variance * b**2 + b_variance * a**2

data = pd.read_csv("test.csv")[["species", "d18_O_w", "d18_O", "d18_O_sd", "d18_O_w_sd"]]

temperatures = np.fromiter(map(predict, data.iterrows()), dtype=float)
standard_deviations = np.fromiter(map(standard_deviation, data.iterrows()), dtype=float)
print(temperatures)
print(standard_deviations)
