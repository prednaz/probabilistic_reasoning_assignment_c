from common import frame_for_stan
import arviz as az
import pandas as pd
import stan

data = pd.read_csv("merged_data.csv")[["temperature", "d18_O_w", "d18_O"]]

posterior = stan.build(
"""\
data {
    int<lower=0> N;
    vector[N] temperature;
    vector[N] d18_O_w;
    vector[N] d18_O;
}
parameters {
    real a;
    real b;
    real<lower=0> sigma;
}
model {
    temperature ~ normal(a + b * (d18_O - d18_O_w), sigma);
    a ~ normal(24, 26);
    b ~ normal(3, 13);
    sigma ~ normal(0, 2);
}
""",
    data=frame_for_stan(data),
    random_seed=1,
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))
