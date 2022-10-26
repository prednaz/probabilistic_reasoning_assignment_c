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
    real a_raw{species};
    real b_raw{species};
"""

def transformed_parameters_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return f"""\
    real a{species} = sigma_a * a_raw{species} + a;
    real b{species} = sigma_b * b_raw{species} + b;
"""

def model_stan(species_data):
    species = to_identifier_suffix(species_data[0])
    return f"""\
    temperature{species} ~ normal(a{species} + b{species} * (d18_O{species} - d18_O_w{species}), sigma);
    a_raw{species} ~ std_normal();
    b_raw{species} ~ std_normal();
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
f"""\
data {{
{"".join(map(data_stan, data))}}}
parameters {{
    real<lower=0> sigma;
    real a;
    real b;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
{"".join(map(parameters_stan, data))}}}
transformed parameters {{
{"".join(map(transformed_parameters_stan, data))}}}
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
#   Gradient evaluation took 0.000176 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 1.76 seconds.
#   Adjust your expectations accordingly!
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Gradient evaluation took 0.000183 seconds
#   1000 transitions using 10 leapfrog steps per transition would take 1.83 seconds.
#   Adjust your expectations accordingly!
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#   Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
#   Exception: normal_lpdf: Location parameter[1] is inf, but must be finite! (in '/run/user/1000/httpstan_vhej3g51/model_6v75qdmh.stan', line 106, column 4 to column 164)
#   If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
#   but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
#                                     mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# sigma                              0.838  0.031   0.779    0.894      0.001    0.000    2578.0    1290.0   1.00
# a                                 15.430  0.889  13.776   17.102      0.040    0.028     491.0     786.0   1.00
# b                                 -3.802  0.247  -4.261   -3.333      0.010    0.007     658.0     907.0   1.01
# sigma_a                            2.888  0.752   1.698    4.268      0.033    0.023     547.0     985.0   1.00
# sigma_b                            0.683  0.238   0.302    1.129      0.009    0.006     661.0     971.0   1.00
# a_raw_cibicides_pachyderma        -0.352  0.320  -0.994    0.222      0.014    0.010     510.0     884.0   1.00
# b_raw_cibicides_pachyderma        -0.878  0.493  -1.858   -0.005      0.018    0.013     760.0    1156.0   1.01
# a_raw_cibicidoides_wuellerstorfi  -1.997  0.556  -3.005   -0.958      0.023    0.016     594.0    1043.0   1.00
# b_raw_cibicidoides_wuellerstorfi   1.260  0.573   0.238    2.351      0.019    0.013     903.0    1204.0   1.00
# a_raw_globorotalia_menardii        0.831  0.444  -0.008    1.708      0.016    0.011     805.0    1043.0   1.00
# b_raw_globorotalia_menardii        0.870  0.574  -0.213    1.891      0.016    0.012    1301.0    1110.0   1.00
# a_raw_hoeglundina_elegans          1.215  0.426   0.404    2.010      0.018    0.013     562.0     847.0   1.00
# b_raw_hoeglundina_elegans         -0.233  0.369  -0.962    0.430      0.014    0.010     709.0    1181.0   1.01
# a_raw_neogloboquadrina_dutertrei   0.944  0.478   0.088    1.861      0.015    0.011    1003.0    1213.0   1.00
# b_raw_neogloboquadrina_dutertrei   0.743  0.587  -0.335    1.848      0.015    0.012    1509.0    1571.0   1.00
# a_raw_orbulina_universa           -0.398  0.458  -1.286    0.420      0.015    0.011     923.0    1254.0   1.00
# b_raw_orbulina_universa           -1.054  0.627  -2.269    0.105      0.017    0.012    1417.0    1583.0   1.00
# a_raw_planulina_ariminensis       -0.628  0.393  -1.330    0.163      0.017    0.012     528.0     750.0   1.00
# b_raw_planulina_ariminensis       -0.083  0.789  -1.536    1.422      0.017    0.019    2113.0    1007.0   1.00
# a_raw_planulina_foveolata         -0.596  0.349  -1.337   -0.013      0.015    0.011     508.0     967.0   1.00
# b_raw_planulina_foveolata         -0.718  0.656  -1.932    0.506      0.016    0.014    1814.0    1080.0   1.00
# a_raw_uvigerina_curticosta         0.580  0.603  -0.534    1.766      0.017    0.013    1296.0    1302.0   1.00
# b_raw_uvigerina_curticosta        -0.416  0.773  -1.742    1.150      0.018    0.015    1764.0    1572.0   1.00
# a_raw_uvigerina_flintii            0.342  0.334  -0.294    0.964      0.014    0.010     556.0     929.0   1.00
# b_raw_uvigerina_flintii            0.673  0.709  -0.602    2.001      0.014    0.014    2537.0    1369.0   1.00
# a_raw_uvigerina_peregrina          0.291  0.323  -0.361    0.868      0.014    0.010     516.0     730.0   1.00
# b_raw_uvigerina_peregrina         -0.354  0.448  -1.173    0.488      0.015    0.011     874.0    1338.0   1.01
# a_cibicides_pachyderma            14.470  0.183  14.129   14.819      0.004    0.003    1879.0    1558.0   1.00
# b_cibicides_pachyderma            -4.352  0.178  -4.678   -4.008      0.004    0.003    2487.0    1332.0   1.00
# a_cibicidoides_wuellerstorfi       9.952  0.682   8.633   11.159      0.015    0.011    2103.0    1324.0   1.00
# b_cibicidoides_wuellerstorfi      -3.013  0.233  -3.417   -2.547      0.005    0.004    2074.0    1400.0   1.00
# a_globorotalia_menardii           17.720  0.856  16.146   19.368      0.017    0.012    2608.0    1414.0   1.00
# b_globorotalia_menardii           -3.253  0.278  -3.761   -2.700      0.005    0.004    2686.0    1444.0   1.00
# a_hoeglundina_elegans             18.732  0.134  18.503   19.002      0.003    0.002    2105.0    1556.0   1.00
# b_hoeglundina_elegans             -3.950  0.054  -4.052   -3.854      0.001    0.001    2098.0    1556.0   1.00
# a_neogloboquadrina_dutertrei      18.047  1.081  15.928   20.005      0.023    0.016    2252.0    1656.0   1.00
# b_neogloboquadrina_dutertrei      -3.317  0.346  -3.983   -2.684      0.007    0.005    2244.0    1715.0   1.00
# a_orbulina_universa               14.315  1.001  12.464   16.300      0.023    0.017    1858.0    1465.0   1.00
# b_orbulina_universa               -4.494  0.381  -5.236   -3.786      0.009    0.006    1826.0    1260.0   1.00
# a_planulina_ariminensis           13.714  0.624  12.552   14.864      0.013    0.009    2343.0    1727.0   1.00
# b_planulina_ariminensis           -3.856  0.523  -4.852   -2.885      0.011    0.008    2218.0    1351.0   1.00
# a_planulina_foveolata             13.806  0.270  13.332   14.330      0.006    0.004    1874.0    1412.0   1.00
# b_planulina_foveolata             -4.286  0.418  -5.123   -3.591      0.009    0.006    2352.0    1419.0   1.00
# a_uvigerina_curticosta            17.057  1.577  14.223   20.363      0.040    0.029    1548.0    1445.0   1.00
# b_uvigerina_curticosta            -4.091  0.522  -5.095   -3.096      0.013    0.010    1536.0    1420.0   1.00
# a_uvigerina_flintii               16.361  0.346  15.690   16.976      0.008    0.005    2141.0    1484.0   1.00
# b_uvigerina_flintii               -3.351  0.481  -4.169   -2.396      0.009    0.007    2555.0    1501.0   1.00
# a_uvigerina_peregrina             16.220  0.308  15.669   16.796      0.006    0.004    2859.0    1595.0   1.00
# b_uvigerina_peregrina             -4.026  0.172  -4.350   -3.712      0.003    0.002    3330.0    1557.0   1.00

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
