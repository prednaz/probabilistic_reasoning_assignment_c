from common import frame_for_stan
import arviz as az
import pandas as pd
import stan

import matplotlib.pyplot as plt

data = pd.read_csv("merged_data.csv")[["temperature", "d18_O_w", "d18_O"]]

posterior = stan.build(
"""\
data {
    int<lower=0> N;
    vector[N] d18_O_w;
    vector[N] d18_O;
}
generated quantities {
    real a = normal_rng(24, 26);
    real b = normal_rng(3, 13);
    real sigma = abs(normal_rng(0, 2));
    array[N] real temperature = normal_rng(a + b * (d18_O - d18_O_w), sigma);
}
""",
    data=frame_for_stan(data),
    random_seed=1,
)
fit = posterior.fixed_param(num_chains=2, num_samples=50)

print(az.summary(fit))

ts_samples = fit["temperature"]
ts_real = data["temperature"]

for ts_sample in ts_samples.T:
    plt.scatter(ts_real, ts_sample)

plt.show()

#                     mean      sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# a                 23.554  52.570  -67.303  110.273      5.366    3.806      90.0      41.0   1.01
# b                 -6.892  18.896  -43.265   24.210      1.501    1.397     144.0      78.0   0.99
# sigma              1.724   1.358    0.074    4.722      0.103    0.079     174.0      78.0   0.99
# temperature[0]    21.765  51.926  -66.703  106.160      5.235    3.713      96.0      66.0   1.01
# temperature[1]    23.310  52.693  -67.334  113.134      5.361    3.802      92.0      66.0   1.01
# ...                  ...     ...      ...      ...        ...      ...       ...       ...    ...
# temperature[372]  49.534  97.317 -123.064  215.641      8.774    7.430     120.0      81.0   1.00
# temperature[373]  49.906  97.552 -120.829  222.036      8.690    7.378     127.0      81.0   1.00
# temperature[374]  49.444  96.778 -119.814  213.032      8.656    7.348     124.0      81.0   1.00
# temperature[375]  49.168  95.941 -117.071  215.085      8.632    7.305     124.0      81.0   1.00
# temperature[376]  48.969  94.448 -115.719  214.546      8.435    7.167     120.0      80.0   1.00

# The effective sample size \texttt{ess_bulk} is less than 400 and the
# Gelman-Rubin statistic \texttt{r_hat} is worse than usually. this is no
# problem because we are not fitting any parameters.
