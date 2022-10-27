from common import (frame_for_stan, to_identifier_suffix)
import arviz as az
from operator import itemgetter
import pandas as pd
import stan

data = pd.read_csv("merged_data.csv")[:1][["species", "temperature", "d18_O_w", "d18_O"]].groupby("species")

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
"""\
data {
    int<lower=0> N_hoeglundina_elegans;
    vector[N_hoeglundina_elegans] temperature_hoeglundina_elegans;
    vector[N_hoeglundina_elegans] d18_O_w_hoeglundina_elegans;
    vector[N_hoeglundina_elegans] d18_O_hoeglundina_elegans;
}
parameters {
    real<lower=0> sigma;
    real a;
    real b;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real a_hoeglundina_elegans;
    real b_hoeglundina_elegans;
}
model {
    print(sigma_a);
    sigma ~ normal(0, 2);
    a ~ normal(24, 26);
    b ~ normal(-3.5, 13.5);
    sigma_a ~ normal(0, 4);
    sigma_b ~ normal(0, 2);
    temperature_hoeglundina_elegans ~ normal(sigma_a * a_hoeglundina_elegans + a + (sigma_b * b_hoeglundina_elegans + b) * (d18_O_hoeglundina_elegans - d18_O_w_hoeglundina_elegans), sigma);
    a_hoeglundina_elegans ~ std_normal();
    b_hoeglundina_elegans ~ std_normal();
}
""",
    data=data_current,
    random_seed=1,
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))
