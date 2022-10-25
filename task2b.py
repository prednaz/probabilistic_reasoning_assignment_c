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
    a ~ normal(17.5, 50);
    b ~ normal(-6.5, 17);
    sigma ~ normal(0, 2);
}
""",
    data={"N": data.shape[0], **data.to_dict("list")},
    random_seed=1,
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))

# Warning in '/run/user/1000/httpstan_51tdyc0k/model_uhosx7i5.stan', line 16, column 21: Argument
#     17 suggests there may be parameters that are not unit scale; consider
#     rescaling with a multiplier (see manual section 22.12).
# Warning in '/run/user/1000/httpstan_51tdyc0k/model_uhosx7i5.stan', line 15, column 21: Argument
#     50 suggests there may be parameters that are not unit scale; consider
#     rescaling with a multiplier (see manual section 22.12).
# Warning in '/run/user/1000/httpstan_51tdyc0k/model_uhosx7i5.stan', line 15, column 15: Argument
#     17.5 suggests there may be parameters that are not unit scale; consider
#     rescaling with a multiplier (see manual section 22.12).
# Sampling: 100% (4000/4000), done.
# Messages received during sampling:
#   Gradient evaluation took 0.000102 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 1.02 seconds.
#   Adjust your expectations accordingly!
#   Gradient evaluation took 9.4e-05 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 0.94 seconds.
#   Adjust your expectations accordingly!
#          mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# a      15.837  0.136  15.577   16.088      0.003    0.002    1536.0    1162.0    1.0
# b      -4.006  0.053  -4.107   -3.911      0.001    0.001    1510.0    1438.0    1.0
# sigma   2.489  0.095   2.312    2.666      0.002    0.002    1578.0    1355.0    1.0

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
