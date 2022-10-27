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
    
    a_avg = model.loc[f"a{species}", "mean"]
    a_std = model.loc[f"a{species}", "sd"]
    a = ufloat(a_avg, a_std)

    b_avg = model.loc[f"b{species}", "mean"]
    b_std = model.loc[f"b{species}", "sd"]
    b = ufloat(b_avg, b_std)

    d18_O_avg = row["d18_O"]
    d18_O_std = row["d18_O_sd"]
    d18_O = ufloat(d18_O_avg, d18_O_std)

    d18_O_w_avg = row["d18_O_w"]
    d18_O_w_std = row["d18_O_w_sd"]
    d18_O_w = ufloat(d18_O_w_avg, d18_O_w_std)

    sigma_T_avg = model.loc["sigma", "mean"] #let's not worry about the error in the error for now
    sigma_T = ufloat(0, sigma_T_avg)

    return a + b * (d18_O - d18_O_w) + sigma_T

data = pd.read_csv("test.csv")[["species", "d18_O_w", "d18_O", "d18_O_sd", "d18_O_w_sd"]]

temperatures = np.fromiter(map(lambda x: predict(x).nominal_value, data.iterrows()), dtype=float)
standard_deviations = np.fromiter(map(lambda x: predict(x).std_dev, data.iterrows()), dtype=float)
print(temperatures)
print(standard_deviations)
