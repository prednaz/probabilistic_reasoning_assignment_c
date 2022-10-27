from common import to_identifier_suffix
from math import sqrt
import numpy as np
import pandas as pd
from uncertainties import ufloat

# `model.csv` contains the ouput of `az.summary` with "p" prepended
model = pd.read_csv("model.csv", sep='\s+', index_col=0)[["mean", "sd"]]

def predict(row):
    row = row[1]
    species = to_identifier_suffix(row["species"])
    
    a = ufloat(
        model.loc[f"a{species}", "mean"],
        model.loc[f"a{species}", "sd"]
    )
    b = ufloat(
        model.loc[f"b{species}", "mean"],
        model.loc[f"b{species}", "sd"]
    )
    d18_O = ufloat(row["d18_O"], row["d18_O_sd"])
    d18_O_w = ufloat(row["d18_O_w"], row["d18_O_w_sd"])
    sigma_T = ufloat(0, model.loc["sigma", "mean"]) #let's not worry about the error in the error for now

    return a + b * (d18_O - d18_O_w) + sigma_T

data = pd.read_csv("test.csv")[["species", "d18_O_w", "d18_O", "d18_O_sd", "d18_O_w_sd"]]

temperatures = np.fromiter(map(lambda x: predict(x).nominal_value, data.iterrows()), dtype=float)
standard_deviations = np.fromiter(map(lambda x: predict(x).std_dev, data.iterrows()), dtype=float)
print(temperatures)
print(standard_deviations)
