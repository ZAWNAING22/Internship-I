import numpy as np

def rmse(I_model, I_exp):
    N = len(I_exp)
    error = I_model - I_exp
    return np.sqrt(np.sum(error**2) / N)

import matplotlib.pyplot as plt

# Experimental (true) current values from RTC France cell (example)
I_exp = np.linspace(0.75, 0.7, 10)  # just for example

# Predicted values by different algorithms (simplified samples)
I_pred_GWO = I_exp + np.random.normal(0, 0.01, size=len(I_exp))
I_pred_SCA = I_exp + np.random.normal(0, 0.005, size=len(I_exp))
I_pred_I_SCHO = I_exp + np.random.normal(0, 0.001, size=len(I_exp))  # best

# Calculate RMSEs using Equation 8
rmse_gwo = rmse(I_pred_GWO, I_exp)
rmse_sca = rmse(I_pred_SCA, I_exp)
rmse_ischo = rmse(I_pred_I_SCHO, I_exp)

# Print the results (like Table 3)
print("GWO RMSE:", rmse_gwo)
print("SCA RMSE:", rmse_sca)
print("I_SCHO RMSE:", rmse_ischo)

# Plot RMSE comparison bar chart
algos = ['GWO', 'SCA', 'I_SCHO']
rmses = [rmse_gwo, rmse_sca, rmse_ischo]

plt.bar(algos, rmses, color=['gray', 'orange', 'green'])
plt.title('RMSE Comparison (based on Equation 8)')
plt.ylabel('RMSE')
plt.grid(axis='y')
plt.show()
