import numpy as np
from scipy.optimize import fsolve

# --- Constants ---
q = 1.602e-19
k = 1.381e-23
T = 306.15  # Temperature in Kelvin
Vt = k * T / q  # Thermal voltage

# === Table 1 (for SDM and PVMM) ===
Vi_table1 = np.array([
    -0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132,
    0.2545, 0.2924, 0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.4590,
    0.4784, 0.4960, 0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736,
    0.5833, 0.5900
])
Ii_table1 = np.array([
    0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7570,
    0.7555, 0.7540, 0.7505, 0.7465, 0.7385, 0.7280, 0.7065, 0.6755,
    0.6320, 0.5730, 0.4990, 0.4130, 0.3165, 0.2120, 0.1035, -0.0100,
    -0.1230, -0.2100
])

# === Table 2 (for DDM) ===
Vi_table2 = np.array([
    -1.9426, 0.1248, 1.8093, 3.3511, 4.7622, 6.0538, 7.2364, 8.3189,
    9.3097, 10.2163, 11.0449, 11.8018, 12.4929, 13.1231, 13.6983,
    14.2221, 14.6995, 15.1346, 15.5311, 15.8929, 16.2229, 16.5241,
    16.7987, 17.0499, 17.2793, 17.4885
])
Ii_table2 = np.array([
    1.0345, 1.0315, 1.0300, 1.0260, 1.0220, 1.0180, 1.0155, 1.0140,
    1.0100, 1.0035, 0.9880, 0.9630, 0.9255, 0.8725, 0.8075,
    0.7265, 0.6345, 0.5345, 0.4275, 0.3185, 0.2085, 0.1010,
    -0.0080, -0.1110, -0.2090, -0.3030
])

# --- Parameters from ALA paper ---
SDM_params = {"Iph": 0.76079, "Is": 0.31073e-6, "Rs": 0.03655, "Rsh": 52.89134, "n": 1.47728}
DDM_params = {"Iph": 0.76079, "Is1": 0.50887e-6, "Is2": 1.67851e-6, "Rs": 0.03696, "Rsh": 55.32626, "n1": 0.09594, "n2": 1.39687}
PVMM_params = {"Iph": 0.20610, "Is": 0.89712e-6, "Rs": 1.95988, "Rsh": 1976.2485, "n": 16.54426, "Ns": 3, "Np": 5}

# --- Model Solvers ---
def sdm_solver(V, params):
    Iph, Is, Rs, Rsh, n = params.values()
    def eq(I):
        return Iph - Is * (np.exp((V + I * Rs) / (n * Vt)) - 1) - (V + I * Rs) / Rsh - I
    return fsolve(eq, Iph)[0]

def ddm_solver(V, params):
    Iph, Is1, Is2, Rs, Rsh, n1, n2 = params.values()
    def eq(I):
        Vt1, Vt2 = n1 * Vt, n2 * Vt
        Vterm = V + I * Rs
        return Iph - Is1 * (np.exp(np.clip(Vterm / Vt1, -100, 100)) - 1) - Is2 * (np.exp(np.clip(Vterm / Vt2, -100, 100)) - 1) - Vterm / Rsh - I
    return fsolve(eq, Iph)[0]

def pvmm_solver(V, params):
    Iph, Is, Rs, Rsh, n, Ns, Np = params.values()
    def eq(I):
        Vt_eff = n * Vt * Ns
        Vterm = V + (I * Rs / Np)
        return Iph * Np - Is * Np * (np.exp(np.clip(Vterm / Vt_eff, -100, 100)) - 1) - Vterm / Rsh - I
    return fsolve(eq, Iph * Np)[0]

# --- RMSE Calculation ---
def calculate_rmse(V_data, I_data, model_func, params):
    Ical = np.array([model_func(V, params) for V in V_data])
    return np.sqrt(np.mean((I_data - Ical) ** 2))

# --- Compute RMSEs ---
rmse_sdm = calculate_rmse(Vi_table1, Ii_table1, sdm_solver, SDM_params)
rmse_ddm = calculate_rmse(Vi_table2, Ii_table2, ddm_solver, DDM_params)
rmse_pvm = calculate_rmse(Vi_table1, Ii_table1, pvmm_solver, PVMM_params)

# --- Print Results ---
print(f"✅ SDM RMSE:  {rmse_sdm:.8f} (target: 7.72986E-04)")
print(f"✅ DDM RMSE:  {rmse_ddm:.8f} (target: 9.88734E-04)")
print(f"✅ PVMM RMSE: {rmse_pvm:.8f} (target: 2.56791E-03)")
