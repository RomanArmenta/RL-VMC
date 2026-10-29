from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .basis import ho_basis


def make_plots(result, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    xmax = max(6.0 / np.sqrt(cfg.omega), 2.0)
    x = np.linspace(-xmax, xmax, 1200); basis = ho_basis(x, cfg.n_basis, cfg.omega)
    up, down = basis @ result.coefficients_up, basis @ result.coefficients_down
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(cfg.n_up): ax.plot(x, up[:, i], label=f"up {i}")
    for i in range(cfg.n_down): ax.plot(x, down[:, i], "--", label=f"down {i}")
    ax.set(xlabel="x", ylabel="orbital", title="Occupied UHF orbitals"); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(out / "orbitals.png", dpi=180); plt.close(fig)
    rho_u = np.sum(up[:, :cfg.n_up]**2, axis=1); rho_d = np.sum(down[:, :cfg.n_down]**2, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(x, rho_u, label="up"); ax.plot(x, rho_d, label="down")
    ax.plot(x, rho_u+rho_d, "k--", label="total"); ax.set(xlabel="x", ylabel="density", title="Spin densities"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "spin_densities.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(result.history["iteration"], result.history["energy"])
    ax.set(xlabel="SCF iteration", ylabel="total energy", title="SCF convergence"); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "scf_convergence.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); idx=np.arange(cfg.n_basis)
    ax.plot(idx, result.eigenvalues_up, "o-", label="up"); ax.plot(idx, result.eigenvalues_down, "s--", label="down")
    ax.set(xlabel="orbital index", ylabel="Fock eigenvalue", title="Fock spectrum"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "fock_spectrum.png", dpi=180); plt.close(fig)


def make_casscf_plots(result, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); cfg = result.config
    xmax = max(6.0 / np.sqrt(cfg.omega), 2.0); x = np.linspace(-xmax, xmax, 1200)
    orbitals = ho_basis(x, cfg.n_basis, cfg.omega) @ result.orbital_coefficients
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(cfg.n_core): ax.plot(x, orbitals[:, i], label=f"core {i}")
    for i in range(cfg.n_active_orbitals):
        j = cfg.n_core+i; ax.plot(x, orbitals[:, j], label=f"active {i}")
    ax.set(xlabel="x", ylabel="orbital", title="Optimized CASSCF orbitals"); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(out / "casscf_orbitals.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(result.history["iteration"], result.history["energy"], "o-")
    ax.set(xlabel="orbital iteration", ylabel="CASSCF energy", title="CASSCF convergence"); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "casscf_convergence.png", dpi=180); plt.close(fig)
    order = np.argsort(np.abs(result.ci_coefficients))[::-1][:min(20, len(result.ci_coefficients))]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(np.arange(len(order)), np.abs(result.ci_coefficients[order])**2)
    ax.set(xlabel="leading determinant (ranked)", ylabel="CI weight", title="Leading CAS determinant weights"); fig.tight_layout()
    fig.savefig(out / "ci_weights.png", dpi=180); plt.close(fig)
    _determinant_mosaics(result, x, orbitals, out)


def _occupied_indices(det, norb):
    value = int(det)
    up = [i for i in range(norb) if value & (1 << i)]
    down = [i for i in range(norb) if value & (1 << (norb+i))]
    return up, down


def _determinant_mosaics(result, x, orbitals, out, per_page=12):
    """Plot coefficient-ranked selected determinants, one orbital-overlay tile each."""
    cfg = result.config
    coeff, dets, _, retained_norm = result.selected_ci()
    n_pages = (len(coeff) + per_page - 1) // per_page
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for page in range(n_pages):
        start, stop = page*per_page, min((page+1)*per_page, len(coeff))
        count = stop-start; cols = 3; rows = (count+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.2*rows), squeeze=False, sharex=True, sharey=True)
        for tile, rank in enumerate(range(start, stop)):
            ax = axes.flat[tile]; up, down = _occupied_indices(dets[rank], cfg.n_active_orbitals)
            for core in range(cfg.n_core):
                ax.plot(x, orbitals[:, core], color="0.65", lw=1.2, label=f"core {core} (2e)")
            for i in up:
                j=cfg.n_core+i; ax.plot(x, orbitals[:, j], color=colors[i%10], lw=1.8, label=f"a{i} up")
            for i in down:
                j=cfg.n_core+i; ax.plot(x, orbitals[:, j], "--", color=colors[i%10], lw=1.8, label=f"a{i} down")
            ax.axhline(0, color="0.8", lw=.6)
            ax.set_title(f"#{rank+1}  c={coeff[rank]:+.5f}  weight={coeff[rank]**2:.4f}", fontsize=10)
            ax.text(.02, .04, f"up={up}  down={down}", transform=ax.transAxes, fontsize=8)
            ax.legend(fontsize=7, ncol=2, loc="upper right")
        for tile in range(count, rows*cols): axes.flat[tile].axis("off")
        fig.suptitle(f"Selected CAS determinants ordered by |CI coefficient|\n"
                     f"page {page+1}/{n_pages}; retained full-CI norm={retained_norm:.8f}")
        fig.supxlabel("x"); fig.supylabel("molecular orbital"); fig.tight_layout(rect=(0.02,0.02,1,.94))
        name = "determinant_mosaic.png" if n_pages == 1 else f"determinant_mosaic_{page+1:02d}.png"
        fig.savefig(out/name, dpi=180); plt.close(fig)
