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
    a{species} ~ normal(17.5, 50);
    b{species} ~ normal(-6.5, 17);
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
{"".join(map(parameters_stan, data))}}}
model {{
    sigma ~ normal(0, 2);
{"".join(map(model_stan, data))}}}
""",
    data=data_current,
    random_seed=1,
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))

#                                 mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# sigma                          0.834  0.032   0.772    0.893      0.001    0.000    2778.0    1626.0    1.0
# a_cibicides_pachyderma        14.496  0.182  14.149   14.816      0.004    0.003    2218.0    1748.0    1.0
# b_cibicides_pachyderma        -4.409  0.177  -4.749   -4.075      0.004    0.003    1819.0    1729.0    1.0
# a_cibicidoides_wuellerstorfi   9.189  0.691   7.984   10.591      0.019    0.013    1382.0    1320.0    1.0
# b_cibicidoides_wuellerstorfi  -2.753  0.236  -3.205   -2.317      0.006    0.004    1459.0    1436.0    1.0
# a_globorotalia_menardii       18.416  0.986  16.685   20.355      0.023    0.016    1826.0    1354.0    1.0
# b_globorotalia_menardii       -3.027  0.321  -3.623   -2.432      0.007    0.005    1836.0    1361.0    1.0
# a_hoeglundina_elegans         18.742  0.142  18.467   19.003      0.003    0.002    1689.0    1328.0    1.0
# b_hoeglundina_elegans         -3.953  0.057  -4.059   -3.848      0.001    0.001    1635.0    1413.0    1.0
# a_neogloboquadrina_dutertrei  19.360  1.228  17.211   21.755      0.030    0.022    1631.0    1407.0    1.0
# b_neogloboquadrina_dutertrei  -2.896  0.393  -3.570   -2.106      0.010    0.007    1635.0    1466.0    1.0
# a_orbulina_universa           13.133  1.218  11.001   15.515      0.033    0.023    1347.0    1302.0    1.0
# b_orbulina_universa           -4.958  0.465  -5.829   -4.086      0.012    0.009    1422.0    1211.0    1.0
# a_planulina_ariminensis       13.585  0.902  11.947   15.234      0.022    0.015    1752.0    1600.0    1.0
# b_planulina_ariminensis       -3.765  0.794  -5.209   -2.274      0.020    0.014    1583.0    1601.0    1.0
# a_planulina_foveolata         13.742  0.265  13.274   14.281      0.005    0.004    2599.0    1422.0    1.0
# b_planulina_foveolata         -4.587  0.508  -5.542   -3.618      0.010    0.007    2632.0    1494.0    1.0
# a_uvigerina_curticosta        22.183  3.413  16.186   28.739      0.083    0.059    1711.0    1354.0    1.0
# b_uvigerina_curticosta        -5.791  1.132  -8.093   -3.887      0.027    0.020    1723.0    1179.0    1.0
# a_uvigerina_flintii           16.339  0.341  15.681   16.943      0.006    0.004    3111.0    1659.0    1.0
# b_uvigerina_flintii           -2.927  0.611  -4.124   -1.841      0.011    0.008    2873.0    1666.0    1.0
# a_uvigerina_peregrina         16.261  0.317  15.673   16.868      0.008    0.005    1698.0    1399.0    1.0
# b_uvigerina_peregrina         -4.048  0.177  -4.390   -3.723      0.004    0.003    1695.0    1428.0    1.0

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
