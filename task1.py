import arviz as az
import pandas as pd
import stan

data = pd.read_csv("merged_data.csv")[["temperature", "d18_O_w", "d18_O"]] # "species",

posterior = stan.build(
"""
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
}
""",
    data={"N": data.shape[0], **data.to_dict("list")},
    random_seed=1
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))

#          mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# a      15.844  0.139  15.578   16.107      0.004    0.003    1465.0    1268.0    1.0
# b      -4.008  0.055  -4.110   -3.901      0.001    0.001    1566.0    1509.0    1.0
# sigma   2.493  0.094   2.322    2.674      0.002    0.002    1545.0    1614.0    1.0

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
