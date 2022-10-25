import arviz as az
import pandas as pd
import stan

data = pd.read_csv("merged_data.csv")[["species", "temperature", "d18_O_w", "d18_O"]]

def stan_input(species_data):
    species = "_" + species_data[0].lower().replace(" ", "_")
    data = {
        f"N{species}": species_data[1].shape[0],
        f"temperature{species}": species_data[1]["temperature"],
        f"d18_O_w{species}": species_data[1]["d18_O_w"],
        f"d18_O{species}": species_data[1]["d18_O"],
    }
    return (
f"""\
    int<lower=0> N{species};
    vector[N{species}] temperature{species};
    vector[N{species}] d18_O_w{species};
    vector[N{species}] d18_O{species};
""",
f"""\
    real a{species};
    real b{species};
    real<lower=0> sigma{species};
""",
f"""\
    temperature{species} ~ normal(a{species} + b{species} * (d18_O{species} - d18_O_w{species}), sigma{species});
    a{species} ~ normal(17.5, 50);
    b{species} ~ normal(-6.5, 17);
    sigma{species} ~ normal(0, 2);
""",
data,
    )

# map(stan_input, data.groupby("species"))

# posterior = stan.build(
# """
# data {
#     int<lower=0> N;
#     vector[N] temperature;
#     vector[N] d18_O_w;
#     vector[N] d18_O;
# }
# parameters {
#     real a;
#     real b;
#     real<lower=0> sigma;
# }
# model {
#     temperature ~ normal(a + b * (d18_O - d18_O_w), sigma);
#     a ~ normal(17.5, 50);
#     b ~ normal(-6.5, 17);
#     sigma ~ normal(0, 2);
# }
# """,
#     data={"N": data.shape[0], **data.to_dict("list")},
#     random_seed=1
# )
# fit = posterior.sample(num_chains=2, num_samples=1000)

# print(az.summary(fit))

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
