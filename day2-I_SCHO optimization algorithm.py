import numpy as np
import matplotlib.pyplot as plt


# --- Single Diode Model Function ---
def single_diode_model(V, Ipv, Io, n, Rs, Rp, T=298.15):
    q = 1.602e-19  # charge of electron
    k = 1.381e-23  # Boltzmann constant
    I = []
    for v in V:
        i = Ipv  # initial guess
        for _ in range(50):
            exp_term = np.exp((q * (v + i * Rs)) / (n * k * T))
            f = Ipv - Io * (exp_term - 1) - (v + i * Rs) / Rp - i
            df = -Io * (q * Rs) / (n * k * T) * exp_term - Rs / Rp - 1
            i = i - f / df
        I.append(i)
    return np.array(I)


# --- Fitness Function (RMSE) ---
def fitness(params, V_exp, I_exp):
    Ipv, Io, n, Rs, Rp = params
    I_model = single_diode_model(V_exp, Ipv, Io, n, Rs, Rp)
    rmse = np.sqrt(np.mean((I_model - I_exp) ** 2))
    return rmse


# --- Simplified I_SCHO Optimizer ---
def i_scho_optimize(fitness_func, bounds, V_exp, I_exp, population=30, max_iter=100):
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    pop = np.random.rand(population, dim) * (ub - lb) + lb
    best = min(pop, key=lambda x: fitness_func(x, V_exp, I_exp))
    best_score = fitness_func(best, V_exp, I_exp)

    for t in range(max_iter):
        for i in range(population):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            W = np.cosh(r1) + np.sinh(r2)
            new_pos = best + r1 * W * (pop[i] - best)
            new_pos = np.clip(new_pos, lb, ub)
            if fitness_func(new_pos, V_exp, I_exp) < fitness_func(pop[i], V_exp, I_exp):
                pop[i] = new_pos
        candidate_best = min(pop, key=lambda x: fitness_func(x, V_exp, I_exp))
        candidate_score = fitness_func(candidate_best, V_exp, I_exp)
        if candidate_score < best_score:
            best = candidate_best
            best_score = candidate_score

    return best, best_score


# --- Main Execution ---
# Simulated voltage
V_exp = np.linspace(0, 0.6, 50)
# Real parameters (to generate experimental current)
I_real_params = [5.5, 1e-10, 1.2, 0.01, 100]
I_exp = single_diode_model(V_exp, *I_real_params)

# Define parameter bounds: [Ipv, Io, n, Rs, Rp]
bounds = [(0, 10), (1e-12, 1e-6), (1, 2), (0, 1), (1, 200)]

# Run optimizer
best_params, best_rmse = i_scho_optimize(fitness, bounds, V_exp, I_exp)

# Output result
print("Best Parameters Found:", best_params)
print("Best RMSE:", best_rmse)

# Plot I-V curve
I_best_fit = single_diode_model(V_exp, *best_params)
plt.plot(V_exp, I_exp, label='Experimental Data')
plt.plot(V_exp, I_best_fit, '--', label='Model Fit')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Single Diode Model Fit using I_SCHO')
plt.legend()
plt.grid()
plt.show()
