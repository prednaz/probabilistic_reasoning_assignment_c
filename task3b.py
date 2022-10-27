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
    real a_raw{species};
    real b_raw{species};
"""

def transformed_parameters_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return f"""\
    real a{species} = sigma_a * a_raw{species} + a;
    real b{species} = sigma_b * b_raw{species} + b;
"""

def model_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return f"""\
    temperature{species} ~ normal(a{species} + b{species} * (d18_O{species} - d18_O_w{species}), sigma);
    a_raw{species} ~ std_normal();
    b_raw{species} ~ std_normal();
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
    real a;
    real b;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
{"".join(map(parameters_stan, data))}}}
transformed parameters {{
{"".join(map(transformed_parameters_stan, data))}}}
model {{
    sigma ~ normal(0, 2);
    a ~ normal(24, 26);
    b ~ normal(-3.5, 13.5);
    sigma_a ~ normal(0, 4);
    sigma_b ~ normal(0, 2);
{"".join(map(model_stan, data))}}}
""",
# sigma_x standard deviations were chosen as
# np.concatenate((fit["a_cibicides_pachyderma"],fit["a_cibicidoides_wuellerstorfi"],fit["a_globorotalia_menardii"],fit["a_hoeglundina_elegans"],fit["a_neogloboquadrina_dutertrei"],fit["a_orbulina_universa"],fit["a_planulina_ariminensis"],fit["a_planulina_foveolata"],fit["a_uvigerina_curticosta"],fit["a_uvigerina_flintii"],fit["a_uvigerina_peregrina"]), axis=None).std(ddof=1)
# and
# np.concatenate((fit["b_cibicides_pachyderma"],fit["b_cibicidoides_wuellerstorfi"],fit["b_globorotalia_menardii"],fit["b_hoeglundina_elegans"],fit["b_neogloboquadrina_dutertrei"],fit["b_orbulina_universa"],fit["b_planulina_ariminensis"],fit["b_planulina_foveolata"],fit["b_uvigerina_curticosta"],fit["b_uvigerina_flintii"],fit["b_uvigerina_peregrina"]), axis=None).std(ddof=1)
    data=data_current,
    random_seed=1,
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))
