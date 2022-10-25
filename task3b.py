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
    temperature{species} ~ normal(sigma_a * a{species} + a + (sigma_b * b{species} + b) * (d18_O{species} - d18_O_w{species}), sigma);
    a{species} ~ std_normal();
    b{species} ~ std_normal();
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

# Messages received during sampling:
#   Gradient evaluation took 0.000194 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 1.94 seconds.
#   Adjust your expectations accordingly!
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Gradient evaluation took 0.000177 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 1.77 seconds.
#   Adjust your expectations accordingly!
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_8x23x98b/model_omke7kqd.stan', line 83, column 4 to column 194)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#                                 mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# sigma                          0.838  0.031   0.780    0.897      0.001    0.000    2764.0    1314.0    1.0
# a                             15.476  0.915  13.696   17.242      0.037    0.026     614.0     812.0    1.0
# b                             -3.822  0.249  -4.262   -3.334      0.009    0.007     723.0     827.0    1.0
# sigma_a                        2.930  0.798   1.619    4.329      0.031    0.022     670.0     806.0    1.0
# sigma_b                        0.678  0.224   0.311    1.101      0.008    0.006     722.0    1156.0    1.0
# a_cibicides_pachyderma        -0.365  0.325  -0.972    0.256      0.013    0.009     668.0     857.0    1.0
# b_cibicides_pachyderma        -0.858  0.501  -1.822    0.020      0.016    0.012     929.0    1024.0    1.0
# a_cibicidoides_wuellerstorfi  -2.002  0.567  -3.124   -1.021      0.020    0.014     778.0    1054.0    1.0
# b_cibicidoides_wuellerstorfi   1.290  0.546   0.246    2.277      0.018    0.012     960.0    1274.0    1.0
# a_globorotalia_menardii        0.791  0.450  -0.022    1.596      0.015    0.010     946.0    1320.0    1.0
# b_globorotalia_menardii        0.862  0.560  -0.175    1.932      0.016    0.012    1177.0    1483.0    1.0
# a_hoeglundina_elegans          1.187  0.432   0.349    1.938      0.017    0.012     666.0     698.0    1.0
# b_hoeglundina_elegans         -0.207  0.373  -0.866    0.490      0.014    0.010     751.0     894.0    1.0
# a_neogloboquadrina_dutertrei   0.922  0.490   0.054    1.879      0.015    0.011    1070.0    1405.0    1.0
# b_neogloboquadrina_dutertrei   0.784  0.606  -0.327    1.958      0.016    0.012    1364.0    1414.0    1.0
# a_orbulina_universa           -0.409  0.466  -1.273    0.456      0.015    0.011     961.0    1221.0    1.0
# b_orbulina_universa           -1.038  0.635  -2.191    0.183      0.016    0.012    1476.0    1342.0    1.0
# a_planulina_ariminensis       -0.638  0.388  -1.416    0.057      0.014    0.010     778.0     786.0    1.0
# b_planulina_ariminensis       -0.066  0.776  -1.525    1.293      0.016    0.016    2218.0    1607.0    1.0
# a_planulina_foveolata         -0.608  0.348  -1.301    0.030      0.013    0.009     693.0     956.0    1.0
# b_planulina_foveolata         -0.684  0.673  -1.962    0.537      0.016    0.014    1844.0    1343.0    1.0
# a_uvigerina_curticosta         0.569  0.589  -0.459    1.711      0.017    0.013    1191.0    1108.0    1.0
# b_uvigerina_curticosta        -0.419  0.775  -1.899    0.964      0.021    0.016    1307.0    1344.0    1.0
# a_uvigerina_flintii            0.328  0.334  -0.277    0.958      0.013    0.009     686.0    1104.0    1.0
# b_uvigerina_flintii            0.663  0.730  -0.666    2.055      0.016    0.013    2128.0    1319.0    1.0
# a_uvigerina_peregrina          0.274  0.331  -0.320    0.921      0.013    0.009     697.0    1081.0    1.0
# b_uvigerina_peregrina         -0.329  0.433  -1.150    0.438      0.014    0.010     931.0    1200.0    1.0

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
