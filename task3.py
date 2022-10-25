import arviz as az
from operator import itemgetter
import pandas as pd
import stan

data = pd.read_csv("merged_data.csv")[["species", "temperature", "d18_O_w", "d18_O"]]

def stan_input(species_data):
    species = "_" + species_data[0].lower().replace(" ", "_")
    data = {
        f"N{species}": species_data[1].shape[0],
        f"temperature{species}": species_data[1]["temperature"].to_list(),
        f"d18_O_w{species}": species_data[1]["d18_O_w"].to_list(),
        f"d18_O{species}": species_data[1]["d18_O"].to_list(),
    }
    return (
f"""\
    int<lower=0> N{species};
    vector[N{species}] temperature{species};
    vector[N{species}] d18_O_w{species};
    vector[N{species}] d18_O{species};
""",
f"""\
    real a{species};
    real b{species};
    real<lower=0> sigma{species};
""",
f"""\
    temperature{species} ~ normal(a{species} + b{species} * (d18_O{species} - d18_O_w{species}), sigma{species});
    a{species} ~ normal(17.5, 50);
    b{species} ~ normal(-6.5, 17);
    sigma{species} ~ normal(0, 2);
""",
        data,
    )

stan_inputs = tuple(map(stan_input, data.groupby("species")))
data_stan = "".join(map(itemgetter(0), stan_inputs))
parameters_stan = "".join(map(itemgetter(1), stan_inputs))
model_stan = "".join(map(itemgetter(2), stan_inputs))

data_current = {}
for data_new in map(itemgetter(3), stan_inputs):
    data_current |= data_new

posterior = stan.build(
f"""
data {{
{data_stan}}}
parameters {{
{parameters_stan}}}
model {{
{model_stan}}}
""",
    data=data_current,
    random_seed=1,
)
fit = posterior.sample(num_chains=2, num_samples=1000)

print(az.summary(fit))

#                                     mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# a_cibicides_pachyderma            14.492  0.138  14.228   14.738      0.003    0.002    1904.0    1540.0   1.00
# b_cibicides_pachyderma            -4.403  0.139  -4.679   -4.168      0.003    0.002    1946.0    1530.0   1.00
# sigma_cibicides_pachyderma         0.629  0.098   0.471    0.814      0.002    0.002    2429.0    1396.0   1.00
# a_cibicidoides_wuellerstorfi       9.202  0.287   8.660    9.714      0.008    0.006    1232.0    1211.0   1.00
# b_cibicidoides_wuellerstorfi      -2.756  0.098  -2.923   -2.564      0.003    0.002    1224.0    1137.0   1.00
# sigma_cibicidoides_wuellerstorfi   0.345  0.027   0.296    0.396      0.001    0.000    1879.0    1293.0   1.01
# a_globorotalia_menardii           18.464  1.168  16.215   20.581      0.034    0.024    1216.0    1303.0   1.00
# b_globorotalia_menardii           -3.012  0.379  -3.748   -2.313      0.011    0.008    1211.0    1258.0   1.00
# sigma_globorotalia_menardii        0.969  0.130   0.748    1.222      0.003    0.002    1702.0    1304.0   1.00
# a_hoeglundina_elegans             18.746  0.165  18.448   19.059      0.004    0.003    1551.0    1474.0   1.00
# b_hoeglundina_elegans             -3.955  0.065  -4.070   -3.827      0.002    0.001    1505.0    1436.0   1.00
# sigma_hoeglundina_elegans          0.983  0.068   0.861    1.105      0.002    0.001    2024.0    1421.0   1.00
# a_neogloboquadrina_dutertrei      19.358  1.278  16.907   21.695      0.038    0.027    1158.0     960.0   1.00
# b_neogloboquadrina_dutertrei      -2.896  0.408  -3.613   -2.084      0.012    0.009    1159.0     977.0   1.00
# sigma_neogloboquadrina_dutertrei   0.798  0.102   0.626    0.992      0.003    0.002    1773.0    1138.0   1.00
# a_orbulina_universa               13.052  1.625  10.162   16.179      0.050    0.035    1089.0     932.0   1.00
# b_orbulina_universa               -4.984  0.621  -6.045   -3.688      0.019    0.014    1083.0     983.0   1.00
# sigma_orbulina_universa            1.093  0.280   0.646    1.615      0.008    0.006    1607.0    1257.0   1.00
# a_planulina_ariminensis           13.606  0.803  12.143   15.074      0.024    0.017    1330.0     799.0   1.00
# b_planulina_ariminensis           -3.772  0.716  -5.123   -2.556      0.022    0.016    1241.0     882.0   1.00
# sigma_planulina_ariminensis        0.669  0.218   0.351    1.068      0.007    0.005    1250.0    1170.0   1.00
# a_planulina_foveolata             13.717  0.541  12.705   14.747      0.013    0.009    1728.0     872.0   1.00
# b_planulina_foveolata             -4.611  1.016  -6.458   -2.726      0.024    0.017    1913.0    1416.0   1.00
# sigma_planulina_foveolata          1.615  0.416   0.949    2.383      0.011    0.008    1556.0    1345.0   1.00
# a_uvigerina_curticosta            22.269  1.981  18.292   25.835      0.059    0.042    1158.0    1149.0   1.00
# b_uvigerina_curticosta            -5.820  0.658  -7.087   -4.603      0.020    0.014    1153.0    1134.0   1.00
# sigma_uvigerina_curticosta         0.468  0.165   0.240    0.757      0.005    0.003    1526.0    1208.0   1.00
# a_uvigerina_flintii               16.329  0.482  15.392   17.273      0.013    0.009    1503.0     977.0   1.00
# b_uvigerina_flintii               -2.944  0.881  -4.812   -1.435      0.023    0.017    1679.0    1078.0   1.00
# sigma_uvigerina_flintii            1.069  0.461   0.438    1.880      0.015    0.011    1103.0    1278.0   1.00
# a_uvigerina_peregrina             16.269  0.474  15.341   17.132      0.013    0.009    1369.0    1284.0   1.00
# b_uvigerina_peregrina             -4.062  0.262  -4.544   -3.577      0.007    0.005    1405.0    1294.0   1.00
# sigma_uvigerina_peregrina          1.275  0.192   0.971    1.662      0.005    0.003    1969.0    1557.0   1.00

# Gelman-Rubin statistic \texttt{r_hat} within 0.05 of 1 indicates that the
# chain has converged and therefore the sample is drawn from the
# posterior. % to do. how do i avoid plagiarizing https://mc-stan.org/docs/cmdstan-guide/stansummary.html ? what to say about effective sample size \texttt{ess_bulk}?
