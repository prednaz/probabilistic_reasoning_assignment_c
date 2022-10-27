from common import to_identifier_suffix
import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy

import matplotlib.pyplot as plt

# `model.csv` contains the ouput of `az.summary` with "p" prepended
model = pd.read_csv("model.csv", sep='\s+', index_col=0)[["mean", "sd"]]

def regress(species, x): 
    species = to_identifier_suffix(species)

    a_avg = model.loc[f"a{species}", "mean"]
    a_std = model.loc[f"a{species}", "sd"]
    a = ufloat(a_avg, a_std)

    b_avg = model.loc[f"b{species}", "mean"]
    b_std = model.loc[f"b{species}", "sd"]
    b = ufloat(b_avg, b_std)
    
    return a + b * x


species = "Hoeglundina elegans"
d = np.linspace(-6, 6, num=50)
T = regress(species, d)

fig, ax = plt.subplots()

ax.plot(d, unumpy.nominal_values(T), "b-", linewidth=0.5)
ax.plot(d, unumpy.nominal_values(T) + 2 * unumpy.std_devs(T), "b--", linewidth=0.5)
ax.plot(d, unumpy.nominal_values(T) - 2 * unumpy.std_devs(T), "b--", linewidth=0.5)

ax.tick_params(axis='both', which='minor', bottom=False)
ax.grid(True, which='minor', axis='both')

ax.set_xlabel("$\delta^{18}O_c - \delta^{18}O_w$")
ax.set_ylabel("$T (^oC)$, 95% confidence interval")

ax.set_title("Regression line: " + species)

fig.savefig("task5.png")
