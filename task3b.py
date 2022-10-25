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
    a{species} ~ normal(a, sigma_a);
    b{species} ~ normal(b, sigma_b);
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
f"""
data {{
{"".join(map(data_stan, data))}}}
parameters {{
    real<lower=0> sigma;
    real a;
    real b;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
{"".join(map(parameters_stan, data))}}}
model {{
    sigma ~ normal(0, 2);
    a ~ normal(17.5, 50);
    b ~ normal(-6.5, 17);
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

#                                 mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# sigma                          0.837  0.033   0.781    0.901      0.001    0.000    2275.0    1420.0   1.00
# a                             15.500  0.935  13.782   17.216      0.019    0.014    2671.0    1640.0   1.00
# b                             -3.804  0.258  -4.303   -3.337      0.006    0.004    2176.0    1549.0   1.00
# sigma_a                        2.934  0.743   1.717    4.309      0.017    0.013    2043.0    1610.0   1.00
# sigma_b                        0.703  0.235   0.289    1.122      0.007    0.005    1128.0    1122.0   1.00
# a_cibicides_pachyderma        14.473  0.187  14.120   14.820      0.004    0.003    2729.0    1404.0   1.00
# b_cibicides_pachyderma        -4.364  0.181  -4.722   -4.047      0.004    0.003    2417.0    1362.0   1.00
# a_cibicidoides_wuellerstorfi   9.896  0.671   8.693   11.237      0.016    0.011    1842.0    1318.0   1.00
# b_cibicidoides_wuellerstorfi  -2.994  0.229  -3.441   -2.574      0.005    0.004    1828.0    1347.0   1.00
# a_globorotalia_menardii       17.722  0.880  16.144   19.452      0.023    0.016    1571.0     976.0   1.00
# b_globorotalia_menardii       -3.253  0.287  -3.769   -2.703      0.007    0.005    1608.0    1078.0   1.00
# a_hoeglundina_elegans         18.728  0.135  18.479   18.988      0.003    0.002    2548.0    1694.0   1.00
# b_hoeglundina_elegans         -3.948  0.052  -4.051   -3.855      0.001    0.001    2574.0    1527.0   1.00
# a_neogloboquadrina_dutertrei  18.098  1.117  15.968   20.160      0.026    0.018    1882.0    1399.0   1.00
# b_neogloboquadrina_dutertrei  -3.300  0.357  -3.985   -2.642      0.008    0.006    1870.0    1264.0   1.00
# a_orbulina_universa           14.278  0.973  12.548   16.128      0.022    0.016    1958.0    1302.0   1.00
# b_orbulina_universa           -4.509  0.371  -5.241   -3.874      0.009    0.006    1847.0    1263.0   1.00
# a_planulina_ariminensis       13.687  0.641  12.489   14.828      0.013    0.009    2386.0    1287.0   1.00
# b_planulina_ariminensis       -3.850  0.534  -4.792   -2.798      0.011    0.008    2193.0    1429.0   1.00
# a_planulina_foveolata         13.798  0.264  13.310   14.284      0.005    0.004    2620.0    1579.0   1.01
# b_planulina_foveolata         -4.309  0.408  -5.096   -3.581      0.009    0.006    2326.0    1634.0   1.00
# a_uvigerina_curticosta        17.098  1.645  13.862   20.080      0.040    0.029    1730.0    1149.0   1.00
# b_uvigerina_curticosta        -4.106  0.548  -5.216   -3.142      0.013    0.009    1777.0    1176.0   1.00
# a_uvigerina_flintii           16.376  0.344  15.732   17.020      0.006    0.004    3099.0    1612.0   1.00
# b_uvigerina_flintii           -3.346  0.490  -4.298   -2.486      0.009    0.007    2788.0    1591.0   1.00
# a_uvigerina_peregrina         16.223  0.297  15.658   16.766      0.006    0.004    2435.0    1530.0   1.00
# b_uvigerina_peregrina         -4.031  0.159  -4.317   -3.727      0.003    0.002    2379.0    1350.0   1.00

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
