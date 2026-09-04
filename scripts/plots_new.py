import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# DATOS DEL BARRIDO EXTENDIDO (16 Ciclos, Recompensa Logarítmica)
# =====================================================================
amplitudes = np.array([
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.020
])

# Energías locales extraídas de tus logs
e_td3 = np.array([
    -65.01, -86.82, 84.94, 119.54, 46.22, 96.45, 43.87, 28.46, 33.88, -1081.61, 46.24, 64.88, -3545.17
])

# Tiempos de autocorrelación (todos colapsados al mínimo matemático ~1.0)
tau_td3 = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00
])

# Muestras útiles (N_eff, maxeadas por la oscilación caótica)
neff_td3 = np.array([
    50000, 50000, 50000, 50000, 50000, 50000, 50000, 49736, 50000, 50000, 50000, 50000, 50000
])

# Muestras por segundo (ESS/sec)
ess_sec_td3 = np.array([
    3755.15, 3756.71, 3765.23, 3759.47, 3762.72, 3762.54, 3757.03, 3744.10, 3763.32, 3758.27, 3747.33, 3757.51, 3762.01
])

# Baselines de MALA colapsado (promediados de tus logs)
e_mala = 40.419
tau_mala = 1.00
neff_mala = 50000
ess_sec_mala = 10860.0

# =====================================================================
# GRAFICACIÓN
# =====================================================================
sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(r"Colapso Físico por Recompensa Logarítmica y Sobreentrenamiento ($16$ ciclos, $N=10, g=5.0$)", 
             fontsize=16, color='darkred', weight='bold')

# --- Panel 1: Energía Local Divergente ---
ax = axs[0, 0]
ax.axhline(e_mala, color='gray', linestyle='--', label=f'MALA Colapsado ({e_mala:.1f} Ha)')
ax.plot(amplitudes, e_td3, 'r-o', label='TD3 (Divergencia)')
ax.set_xscale('log')
ax.set_yscale('symlog', linthresh=50) 
ax.set_title("Energía Local (Evidencia de ruptura nodal)")
ax.set_ylabel("Hartrees (SymLog)")
ax.axhspan(-10, 34, color='green', alpha=0.1, label='Rango Físico Esperado (~32 Ha)')
ax.legend(loc='lower left')

# --- Panel 2: Falsa Autocorrelación ---
ax = axs[0, 1]
ax.axhline(tau_mala, color='gray', linestyle='--', label='MALA')
ax.plot(amplitudes, tau_td3, 'm-o', label='TD3')
ax.set_xscale('log')
ax.set_ylim(0.5, 5.0)
ax.set_title(r"Tiempo de Autocorrelación $\tau_{int}$ (Ruido Blanco)")
ax.set_ylabel(r"$\tau_{int}$")
ax.text(0.003, 2.5, "Falso Positivo:\nLos saltos caóticos arrojan\ncorrelación nula", 
        bbox=dict(facecolor='yellow', alpha=0.3), ha='center')
ax.legend()

# --- Panel 3: Falso N_eff ---
ax = axs[1, 0]
ax.axhline(neff_mala, color='gray', linestyle='--', label='MALA')
ax.plot(amplitudes, neff_td3, 'b-o', label='TD3')
ax.set_xscale('log')
ax.set_ylim(0, 60000)
ax.set_title(r"Muestras Útiles Extraídas ($N_{eff}$)")
ax.set_ylabel("Muestras")
ax.set_xlabel("Actor Amplitude (Log)")
ax.legend()

# --- Panel 4: ESS / sec (Dinámica Atascada) ---
ax = axs[1, 1]
ax.axhline(ess_sec_mala, color='gray', linestyle='--', label='MALA (Pura Fuerza Bruta)')
ax.plot(amplitudes, ess_sec_td3, 'g-o', label='TD3')
ax.set_xscale('log')
ax.set_title("Eficiencia Computacional Neta (ESS / Segundo)")
ax.set_ylabel("Muestras / seg")
ax.set_xlabel("Actor Amplitude (Log)")
ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig("actor_amplitude_collapse.png", dpi=300)
print("Gráfica guardada como 'actor_amplitude_collapse.png'")