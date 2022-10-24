import arviz as az
import pandas as pd
import stan

data = pd.read_csv("merged_data.csv")[["temperature", "d18_O_w", "d18_O"]]

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
    a ~ normal(0, 50);
    b ~ normal(0, 17);
    sigma ~ normal(0, 2);
}
""",
    data={"N": data.shape[0], **data.to_dict("list")},
    random_seed=1
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))

# Warning in '/run/user/1000/httpstan_ve_tz6ss/model_zjesnkki.stan', line 16, column 18: Argument
#     17 suggests there may be parameters that are not unit scale; consider
#     rescaling with a multiplier (see manual section 22.12). % to do. https://mc-stan.org/docs/stan-users-guide/standardizing-predictors-and-outputs.html
# Warning in '/run/user/1000/httpstan_ve_tz6ss/model_zjesnkki.stan', line 15, column 18: Argument
#     50 suggests there may be parameters that are not unit scale; consider
#     rescaling with a multiplier (see manual section 22.12).
# Sampling: 100% (4000/4000), done.
# Messages received during sampling:
#   Gradient evaluation took 0.000107 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 1.07 seconds.
#   Adjust your expectations accordingly!
#   Gradient evaluation took 9.2e-05 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 0.92 seconds.
#   Adjust your expectations accordingly!
#          mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# a      15.846  0.142  15.589   16.114      0.004    0.003    1487.0    1366.0    1.0
# b      -4.009  0.055  -4.119   -3.912      0.001    0.001    1445.0    1496.0    1.0
# sigma   2.487  0.094   2.311    2.660      0.002    0.002    1474.0    1617.0    1.0

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
