import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Datos combinados: Escaneo fino (0.006-0.015) + Escaneo grueso (0.1 - 1.0)
amplitudes = np.array([
    0.001, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
])

# Energías locales
e_td3 = np.array([
    32.569, 32.570, 32.528, 32.309, 32.463, 34.058, 32.589, 32.501, 32.593, 36.073, 33.038,
    32.684, -124.13, 48.209, -1990.7, -1457.5, -11.205, -45.301, -7944.9, -3285.7, -208.48
])

# Tiempos de autocorrelación
tau_td3 = np.array([
    6.26, 5.51, 6.13, 1.00, 4.33, 5.91, 6.82, 5.30, 6.53, 2.15, 4.81,
    16.39, 1.00, 39.36, 319.40, 43.30, 74.70, 101.47, 319.40, 319.40, 27.91
])

# Muestras útiles (N_eff)
neff_td3 = np.array([
    798, 907, 815, 5000, 1154, 845, 732, 944, 766, 2324, 1038,
    305, 5000, 127, 15, 115, 66, 49, 15, 15, 179
])

# Muestras por segundo (ESS/sec)
ess_sec_td3 = np.array([
    333.45, 370.59, 333.59, 2028.86, 467.61, 345.30, 300.32, 383.97, 312.29, 947.34, 432.14,
    125.74, 2051.23, 52.84, 6.48, 47.25, 27.90, 20.54, 6.54, 6.44, 74.72
])

# Constantes MALA baseline
e_mala = 32.785
tau_mala = 20.39
neff_mala = 245
ess_sec_mala = 229.0

sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r"Sensibilidad de RL-VMC ante la Amplitud del Actor ($N=10, g=5.0$)", fontsize=16)

# Puntos válidos vs Colapso (Filtro para las gráficas de eficiencia)
# Marcamos como no válidos los puntos donde la energía divergió a valores negativos o muy altos
valid_mask = e_td3 > 0

# --- Panel 1: Energía Local ---
ax = axs[0, 0]
ax.axhline(e_mala, color='gray', linestyle='--', label='MALA Baseline (32.78 Ha)')
ax.plot(amplitudes, e_td3, 'r-o', label='TD3')
ax.set_xscale('log')
ax.set_yscale('symlog', linthresh=10) # symlog permite ver valores negativos grandes y positivos juntos
ax.set_title("Energía de Medición Final")
ax.set_ylabel("Hartrees (SymLog)")
ax.axvspan(0.15, 1.0, color='red', alpha=0.1, label='Zona de Colapso Físico')
ax.legend()

# --- Panel 2: Tiempo de Autocorrelación (Tau) ---
ax = axs[0, 1]
ax.axhline(tau_mala, color='gray', linestyle='--', label='MALA')
ax.plot(amplitudes[valid_mask], tau_td3[valid_mask], 'g-o', label='TD3 (Física Válida)')
ax.plot(amplitudes[~valid_mask], tau_td3[~valid_mask], 'r-x', label='TD3 (Colapso)', alpha=0.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(r"Tiempo de Autocorrelación ($\tau_{int}$) - Menor es mejor")
ax.set_ylabel(r"$\tau_{int}$")
ax.legend()

# --- Panel 3: Tamaño de Muestra Efectivo (N_eff) ---
ax = axs[1, 0]
ax.axhline(neff_mala, color='gray', linestyle='--', label='MALA')
ax.plot(amplitudes[valid_mask], neff_td3[valid_mask], 'b-o', label='TD3 (Física Válida)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Muestras Útiles Extraídas ($N_{eff}$) - Mayor es mejor")
ax.set_ylabel("Muestras")
ax.set_xlabel("Actor Amplitude (Log)")
ax.legend()

# --- Panel 4: Eficiencia Computacional (ESS / seg) ---
ax = axs[1, 1]
ax.axhline(ess_sec_mala, color='gray', linestyle='--', label='MALA')
ax.plot(amplitudes[valid_mask], ess_sec_td3[valid_mask], 'm-o', label='TD3 (Física Válida)')

# Anotar el punto óptimo
# Anotar el punto óptimo
opt_idx = 3 # Índice para amp=0.008 (¡Tu nuevo récord!)
ax.annotate(f'Óptimo (Amp=0.008)\n{int(ess_sec_td3[opt_idx])} ESS/s', 
            xy=(amplitudes[opt_idx], ess_sec_td3[opt_idx]), 
            xytext=(0.02, 1000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

ax.set_xscale('log')
ax.set_title("Eficiencia Computacional Neta (ESS / Segundo)")
ax.set_ylabel("Muestras útiles / seg")
ax.set_xlabel("Actor Amplitude (Log)")
ax.legend()

plt.tight_layout()
plt.savefig("actor_amplitude_sweep.png", dpi=300)
print("Gráfica guardada como 'actor_amplitude_sweep.png'")
plt.show()