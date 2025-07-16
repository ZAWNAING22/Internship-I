import pandas as pd
import numpy as np
from scipy.optimize import fsolve

# -----------------------------
# 🔁 Constants (for both models)
# -----------------------------
q = 1.602e-19   # Electron charge (C)
k = 1.381e-23   # Boltzmann constant (J/K)
T = 298         # Temperature (Kelvin)

# ------------------------
# 📌 SDM Model Parameters
# ------------------------
Iph_sdm = 0.8
Isd_sdm = 1e-6
Rs_sdm = 0.01
Rsh_sdm = 100
n_sdm = 1.3

# ------------------------
# 📌 DDM Model Parameters
# ------------------------
Iph_ddm = 0.8
Isd1_ddm = 1e-6
Isd2_ddm = 1e-7
Rs_ddm = 0.01
Rsh_ddm = 100
n1_ddm = 1.3
n2_ddm = 2.0

# ---------------------------------------------
# 📁 Load Excel File (Replace with real name!)
# ---------------------------------------------
excel_path = "your_excel_file.xlsx"  # 🔁 Replace this with your actual file
sheet_sdm = "SDM_Data"               # 🔁 Sheet name for SDM data
sheet_ddm = "DDM_Data"               # 🔁 Sheet name for DDM data

# Read both sheets (Voltage must be a column)
df_sdm = pd.read_excel(excel_path, sheet_name=sheet_sdm)
df_ddm = pd.read_excel(excel_path, sheet_name=sheet_ddm)

# Normalize column names
df_sdm.columns = [col.strip().lower() for col in df_sdm.columns]
df_ddm.columns = [col.strip().lower() for col in df_ddm.columns]

# -----------------------------
# 🧠 Define SDM Equation
# -----------------------------
def sdm_equation(IL, V):
    return Iph_sdm - Isd_sdm * (np.exp(q*(V + Rs_sdm*IL)/(n_sdm*k*T)) - 1) - (V + Rs_sdm*IL)/Rsh_sdm - IL

# -----------------------------
# 🧠 Define DDM Equation
# -----------------------------
def ddm_equation(IL, V):
    term1 = Isd1_ddm * (np.exp(q*(V + Rs_ddm*IL)/(n1_ddm*k*T)) - 1)
    term2 = Isd2_ddm * (np.exp(q*(V + Rs_ddm*IL)/(n2_ddm*k*T)) - 1)
    return Iph_ddm - term1 - term2 - (V + Rs_ddm*IL)/Rsh_ddm - IL

# -----------------------------------------
# 🧪 Apply SDM Calculation for Each Voltage
# -----------------------------------------
df_sdm['il_calculated_sdm'] = df_sdm['voltage'].apply(lambda V: fsolve(sdm_equation, x0=0.5, args=(V))[0])

# -----------------------------------------
# 🧪 Apply DDM Calculation for Each Voltage
# -----------------------------------------
df_ddm['il_calculated_ddm'] = df_ddm['voltage'].apply(lambda V: fsolve(ddm_equation, x0=0.5, args=(V))[0])

# -----------------------------------------
# 💾 Save to Excel with Two Sheets
# -----------------------------------------
output_file = "SDM_DDM_Output.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_sdm.to_excel(writer, sheet_name="SDM_Output", index=False)
    df_ddm.to_excel(writer, sheet_name="DDM_Output", index=False)

print(f"✅ Done. Results saved to: {output_file}")
