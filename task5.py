from common import to_identifier_suffix
import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from matplotlib.colors import hsv_to_rgb

# `model.csv` contains the ouput of `az.summary` with "p" prepended
model = pd.read_csv("model.csv", sep='\s+', index_col=0)[["mean", "sd"]]

data = pd.read_csv("merged_data.csv")[["species"]]

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

ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.grid(True, which='both')

ax.set_xlabel("$\delta^{18}O_c - \delta^{18}O_w$")
ax.set_ylabel("$T (^oC)$, 95% confidence interval")

ax.set_title("Regression line: " + species)

fig.savefig("task5-1.png")

species = data["species"]
species = np.unique(species)

n = len(species)

fig, ax = plt.subplots(figsize=(8.4, 6.8))
for i, species in enumerate(species):
    T = regress(species, d)
        
    c = hsv_to_rgb((i / n, 1.0, 1.0))

    ax.plot(d, unumpy.nominal_values(T), "-", linewidth=0.5, c=c, label=species)
    ax.plot(d, unumpy.nominal_values(T) + 2 * unumpy.std_devs(T), "--", linewidth=0.5, c=c)
    ax.plot(d, unumpy.nominal_values(T) - 2 * unumpy.std_devs(T), "--", linewidth=0.5, c=c)

ax.legend(bbox_to_anchor=(1.0,1.0), loc="upper left")
ax.set_xlabel("$\delta^{18}O_c - \delta^{18}O_w$")
ax.set_ylabel("$T (^oC)$, 95% confidence interval")

ax.set_title("Regression lines: all species")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7)
fig.savefig("task5-2.png")
