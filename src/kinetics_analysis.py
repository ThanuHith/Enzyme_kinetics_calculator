import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Michaelis-Menten Equation
# -----------------------------

def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

# -----------------------------
# Load Experimental Data
# -----------------------------

data = pd.read_csv("../data/enzyme_data.csv")

substrate_conc = data["substrate"].values
velocity = data["velocity"].values

# -----------------------------
# Curve Fitting
# -----------------------------

params, covariance = curve_fit(
    michaelis_menten,
    substrate_conc,
    velocity,
    p0=[1.0, 1.0]
)

Vmax, Km = params

print("===== Calculated Enzyme Parameters =====")
print(f"Vmax = {Vmax:.3f} µmol/min")
print(f"Km   = {Km:.3f} mM")

# -----------------------------
# Generate Smooth Curve
# -----------------------------

s_smooth = np.linspace(0, 25, 200)
v_smooth = michaelis_menten(s_smooth, Vmax, Km)

# -----------------------------
# Plot Michaelis-Menten Curve
# -----------------------------

plt.figure()

plt.scatter(substrate_conc, velocity, color="red", label="Experimental Data")
plt.plot(s_smooth, v_smooth, label="Fitted Curve")

plt.xlabel("Substrate Concentration (mM)")
plt.ylabel("Velocity (µmol/min)")
plt.title("Michaelis-Menten Enzyme Kinetics")

plt.legend()
plt.grid(True)

plt.savefig("../results/michaelis_menten_plot.png")

plt.show()

# -----------------------------
# Lineweaver-Burk Plot
# -----------------------------

inv_S = 1 / substrate_conc
inv_v = 1 / velocity

plt.figure()

plt.scatter(inv_S, inv_v)

plt.xlabel("1/[S]")
plt.ylabel("1/V")

plt.title("Lineweaver-Burk Plot")

plt.grid(True)

plt.savefig("../results/lineweaver_burk_plot.png")

plt.show()