import arviz as az
import pandas as pd
import stan

import matplotlib.pyplot as plt

data = pd.read_csv("merged_data.csv")[["temperature", "d18_O_w", "d18_O"]]

posterior = stan.build(
"""
data {
    int<lower=0> N;
    vector[N] d18_O_w;
    vector[N] d18_O;
}
generated quantities {
    real a = normal_rng(17.5, 50);
    real b = normal_rng(-6.5, 17);
    real sigma = abs(normal_rng(0, 2));
    array[N] real temperature = normal_rng(a + b * (d18_O - d18_O_w), sigma);
}
""",
    data={"N": data.shape[0], **data.to_dict("list")},
    random_seed=1
)
fit = posterior.fixed_param(num_chains=2, num_samples=50)

ts_samples = fit["temperature"]
ts_real = data["temperature"]



for ts_sample in ts_samples.T:
    plt.scatter(ts_real, ts_sample)

plt.show()
