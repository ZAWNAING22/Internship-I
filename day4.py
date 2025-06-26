from scipy.stats import friedmanchisquare

# Example RMSE results from 3 algorithms over 5 datasets
rmse_gwo = [0.06, 0.07, 0.08, 0.065, 0.07]
rmse_sca = [0.05, 0.045, 0.048, 0.046, 0.047]
rmse_ischo = [0.03, 0.031, 0.029, 0.032, 0.028]

# Perform Friedman Test
stat, p = friedmanchisquare(rmse_gwo, rmse_sca, rmse_ischo)
print("Friedman Test statistic:", stat)
print("P-value:", p)

from scipy.stats import wilcoxon

# Example RMSE results of SCA vs I_SCHO across 5 datasets
rmse_sca = [0.05, 0.045, 0.048, 0.046, 0.047]
rmse_ischo = [0.03, 0.031, 0.029, 0.032, 0.028]

# Perform Wilcoxon test
stat, p = wilcoxon(rmse_sca, rmse_ischo)
print("Wilcoxon Test statistic:", stat)
print("P-value:", p)
