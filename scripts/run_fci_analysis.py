import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial, roots_hermite

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
from pyscf import fci, lib
from pyscf.fci import addons

def build_1d_ho_integrals(n_basis, nq, omega, g, softening):
    t, weights = roots_hermite(nq)
    x = t / np.sqrt(omega)
    polys = np.stack([eval_hermite(k, t) / np.sqrt(2.0**k * factorial(k)) for k in range(n_basis)], axis=0)
    products = np.einsum("pi,qi->pqi", polys, polys) / np.sqrt(np.pi)
    
    h1e = np.diag((np.arange(n_basis) + 0.5) * omega)
    distance = x[:, None] - x[None, :]
    kernel = g / np.sqrt(distance**2 + softening**2)
    eri = np.einsum("i,j,pqi,rsj,ij->pqrs", weights, weights, products, products, kernel, optimize=True)
    return h1e, eri

def main():
    n_elec = 10
    nelec_tuple = (5, 5) 
    omega = 0.1
    g = 20.0
    softening = 0.1
    
    # Si tienes RAM suficiente (unos 16 GB), súbelo a 14 para ver cómo mejoran los picos.
    # Si se congela, regrésalo a 12.
    n_basis = 12       
    nq = 100           

    print("Calculando integrales...")
    h1e, eri = build_1d_ho_integrals(n_basis, nq, omega, g, softening)

    # 1. INTERCEPTAR EL LOG DE PYSCF
    log_buffer = io.StringIO()
    
    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 50
    cisolver.conv_tol = 1e-8
    cisolver.verbose = 4 
    cisolver.stdout = log_buffer # Forzamos a PySCF a escribir su log interno en nuestra variable
    
    lib.num_threads(os.cpu_count())
    cisolver.max_memory = 60000 

    print("Ejecutando Davidson FCI...")
    e_fci, v_fci = cisolver.kernel(h1e, eri, n_basis, nelec_tuple)

    # 2. PARSEAR LA HISTORIA DE ENERGÍA Y GRAFICARLA
    log_text = log_buffer.getvalue()
    energies = []
    
    for line in log_text.split('\n'):
        if 'cycle' in line and 'E =' in line:
            parts = line.split()
            try:
                idx = parts.index('E')
                energies.append(float(parts[idx + 2]))
            except:
                pass
                
    if energies:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(energies) + 1), energies, marker='o', color='#c0392b', linewidth=2)
        plt.title("Convergencia del Algoritmo de Davidson (Full CI)", weight='bold')
        plt.xlabel("Ciclo (Iteración)")
        plt.ylabel("Energía (Hartree)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("Convergencia_Davidson_FCI.pdf", dpi=300)
        print("\n-> Gráfica de energía guardada como 'Convergencia_Davidson_FCI.pdf'")

    # 3. EXTRAER DETERMINANTES (Umbral más bajo)
    print("\n--- TOP 20 DETERMINANTES MÁS PESADOS ---")
    # Bajamos el umbral a 0.01 para que podamos ver cómo está distribuida la función de onda
    addons.large_ci(v_fci, n_basis, nelec_tuple, tol=0.01)

if __name__ == "__main__":
    main()