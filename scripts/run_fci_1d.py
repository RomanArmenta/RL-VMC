import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial, roots_hermite

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
from pyscf import fci, lib
from pyscf.fci import addons

def build_1d_ho_integrals(n_basis, nq, omega, g, softening):
    """Genera las integrales espaciales 1D exactas usando cuadratura de Gauss-Hermite."""
    t, weights = roots_hermite(nq)
    x = t / np.sqrt(omega)
    
    # Evaluar polinomios
    polys = np.stack([eval_hermite(k, t) / np.sqrt(2.0**k * factorial(k)) for k in range(n_basis)], axis=0)
    products = np.einsum("pi,qi->pqi", polys, polys) / np.sqrt(np.pi)
    
    # 1. Hamiltoniano de un cuerpo (Diagonal exacta para Oscilador Armónico)
    h1e = np.diag((np.arange(n_basis) + 0.5) * omega)
    
    # 2. Integrales de dos cuerpos (Interacción Soft-Coulomb)
    # PySCF espera la notación química: (ij|kl) = \int \phi_i(1)\phi_j(1) V(1,2) \phi_k(2)\phi_l(2)
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
    n_basis = 12       
    nq = 100           

    print(f"===========================================================")
    print(f" Iniciando Full CI (PySCF) para N={n_elec}, g={g}")
    print(f"===========================================================")

    h1e, eri = build_1d_ho_integrals(n_basis, nq, omega, g, softening)

    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-8
    
    # 1. VISUALIZAR LA MINIMIZACIÓN: 
    # Nivel 4 imprime la energía en cada macro-iteración de Davidson
    cisolver.verbose = 4 
    
    lib.num_threads(os.cpu_count())
    cisolver.max_memory = 60000 

    t1 = time.time()
    e_fci, v_fci = cisolver.kernel(h1e, eri, n_basis, nelec_tuple)
    t_total = time.time() - t1
    
    print(f"\n===========================================================")
    print(f" -> Energía Exacta FCI (Base={n_basis}): {e_fci:.6f} Hartree")
    print(f" -> Tiempo de diagonalización: {t_total / 60:.2f} minutos")
    print(f"===========================================================")

    # 2. EXTRAER LOS DETERMINANTES MÁS RELEVANTES
    print("\n--- DETERMINANTES DOMINANTES ---")
    # tol=0.05 mostrará los determinantes con un coeficiente absoluto mayor a 0.05
    addons.large_ci(v_fci, n_basis, nelec_tuple, tol=0.05)

    # 3. EXTRAER ORBITALES NATURALES Y DENSIDAD ESPACIAL
    print("\n--- CALCULANDO MATRIZ DE DENSIDAD ESPACIAL ---")
    # Obtener la 1-RDM en la base del oscilador armónico
    dm1 = cisolver.make_rdm1(v_fci, n_basis, nelec_tuple)
    
    # Diagonalizar la 1-RDM para obtener ocupaciones fraccionarias
    occ, nat_orbs = np.linalg.eigh(dm1)
    idx = np.argsort(occ)[::-1]
    occ = occ[idx]
    
    print("Ocupaciones de los Orbitales Naturales (Si son fraccionarias, hay fuerte correlación):")
    for i, o in enumerate(occ):
        if o > 1e-3: print(f" Orbital {i}: {o:.4f}")

    # 4. GRAFICAR LA DENSIDAD ESPACIAL \rho(x)
    x = np.linspace(-15, 15, 600)
    t_vals = x * np.sqrt(omega)
    
    # Evaluar las funciones de base en la malla espacial
    polys = np.stack([eval_hermite(k, t_vals) / np.sqrt(2.0**k * factorial(k)) for k in range(n_basis)], axis=0)
    basis_vals = (omega / np.pi)**0.25 * np.exp(-0.5 * t_vals**2) * polys
    
    # \rho(x) = \sum_{i,j} dm1_{ij} \phi_i(x) \phi_j(x)
    rho_x = np.einsum('ij,ix,jx->x', dm1, basis_vals, basis_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x, rho_x, color='#2c3e50', linewidth=2, label='Densidad Total FCI')
    plt.fill_between(x, rho_x, color='#34495e', alpha=0.3)
    plt.title(f"Densidad Espacial FCI (N={n_elec}, g={g}, Base={n_basis})", weight='bold')
    plt.xlabel("Posición (Bohr)")
    plt.ylabel(r"$\rho(x)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Densidad_FCI_1D.pdf", dpi=300)
    print("Gráfica guardada como 'Densidad_FCI_1D.pdf'")

if __name__ == "__main__":
    main()