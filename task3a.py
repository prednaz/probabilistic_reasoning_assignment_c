from common import (frame_for_stan, to_identifier_suffix)
import arviz as az
from operator import itemgetter
import pandas as pd
import stan

data = pd.read_csv("merged_data.csv")[["species", "temperature", "d18_O_w", "d18_O"]].groupby("species")

def data_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return f"""\
    int<lower=0> N{species};
    vector[N{species}] temperature{species};
    vector[N{species}] d18_O_w{species};
    vector[N{species}] d18_O{species};
"""

def parameters_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return f"""\
    real a{species};
    real b{species};
"""

def model_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return f"""\
    temperature{species} ~ normal(a{species} + b{species} * (d18_O{species} - d18_O_w{species}), sigma);
    a{species} ~ normal(24, 26);
    b{species} ~ normal(-3.5, 13.5);
"""

def data_for_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return {
        key + species: value
        for (key, value)
        in frame_for_stan(species_data[1][["temperature", "d18_O_w", "d18_O"]]).items()
    }

data_current = {}
for data_new in map(data_for_stan, data):
    data_current.update(data_new)

posterior = stan.build(
f"""\
data {{
{"".join(map(data_stan, data))}}}
parameters {{
    real<lower=0> sigma;
{"".join(map(parameters_stan, data))}}}
model {{
    sigma ~ normal(0, 2);
{"".join(map(model_stan, data))}}}
""",
    data=data_current,
    random_seed=1,
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))
