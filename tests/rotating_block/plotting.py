import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

plt.rcParams.update({
    # "text.usetex": True,           # Benötigt eine lokale LaTeX-Installation
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 400              # Hohe Auflösung für Druck
})


data = np.loadtxt("angular_momentum_data_600.0s.csv", delimiter=",", skiprows=1)


t, L = data[:, 0], data[:, 1]
L_mean = np.mean(L)

fig, ax = plt.subplots(figsize=(6, 4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
ax.plot(t, L, label="Angular Momentum L(t)")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"L [$\text{kg} \cdot \text{m}^2 / \text{s}$]")
ax.set_title("Angular Momentum over Time")
ax.set_ylim([min(L)*0.9, max(L)*1.1])
ax.grid()
# ax.legend()
fig.savefig("angular_momentum_600s_zoomed.png")

plt.clf()

fig, ax = plt.subplots(figsize=(6, 4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

L_relative_log_error = np.log(np.abs((L - L_mean) / L_mean))
ax.plot(t, L_relative_log_error, label="Relative Log Error in L(t)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("$\\log\\left|\\frac{L(t) - \\bar{L}}{\\bar{L}}\\right|$")
ax.set_title("Relative Log Error of Angular Momentum over Time")
ax.grid()
# ax.legend()  
fig.savefig("angular_momentum_log_error_600s.png")
print("Saved angular momentum analysis plot to angular_momentum_log_error_600s.png")