from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _save(fig, path, dpi=180):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_density(density, classical_sites, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = density["x"]
    ax.plot(x, density["rho_total"], label=r"$\rho(x)$")
    ax.plot(x, density["rho_up"], ls="--", label=r"$\rho_\uparrow(x)$")
    ax.plot(x, density["rho_down"], ls=":", label=r"$\rho_\downarrow(x)$")
    for site in np.asarray(classical_sites): ax.axvline(site, lw=0.8, alpha=0.35)
    ax.set_xlabel("x"); ax.set_ylabel("physical one-body density"); ax.set_ylim(ymin=0)
    ax.legend(frameon=False); _save(fig, path)


def _add_harmonic_potential(ax, x, omega):
    potential = 0.5 * float(omega) ** 2 * np.asarray(x) ** 2
    right = ax.twinx()
    right.plot(x, potential, color="0.35", lw=1.4, alpha=0.65,
               label=r"$V_{\rm HO}(x)=\frac{1}{2}\omega^2x^2$")
    right.fill_between(x, 0.0, potential, color="0.75", alpha=0.12)
    right.set_ylabel("harmonic potential")
    right.set_ylim(bottom=0.0)
    return right


def _combined_legend(ax, right, ncol=2):
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False,
              ncol=ncol, fontsize=8)


def plot_density_with_potential(density, omega, classical_sites, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = density["x"]
    ax.plot(x, density["rho_total"], lw=1.8, label=r"$\rho(x)$")
    ax.plot(x, density["rho_up"], ls="--", label=r"$\rho_\uparrow(x)$")
    ax.plot(x, density["rho_down"], ls=":", label=r"$\rho_\downarrow(x)$")
    for site in np.asarray(classical_sites):
        ax.axvline(site, color="0.3", lw=0.7, alpha=0.25)
    ax.set_xlabel("x"); ax.set_ylabel("physical one-body / spin density")
    ax.set_ylim(ymin=0)
    right = _add_harmonic_potential(ax, x, omega)
    _combined_legend(ax, right); _save(fig, path)


def _orbital_count(orbitals, occupied, method):
    return min(orbitals.shape[1], occupied if method == "hf" else 8)


def _plot_orbital_quantity(x, up, down, omega, n_up, n_down, method, path, squared):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.0), sharex=True)
    for ax, values, occupied, spin, style in (
        (axes[0], up, n_up, r"\uparrow", "-"),
        (axes[1], down, n_down, r"\downarrow", "--"),
    ):
        count = _orbital_count(values, occupied, method)
        for i in range(count):
            y = values[:, i] ** 2 if squared else values[:, i]
            label = rf"$|\phi_{{{i+1},{spin}}}|^2$" if squared else rf"$\phi_{{{i+1},{spin}}}$"
            ax.plot(x, y, ls=style, lw=1.35, label=label)
        ax.axhline(0.0, color="0.25", lw=0.6)
        ax.set_ylabel(r"$|\phi_p(x)|^2$" if squared else r"$\phi_p(x)$")
        right = _add_harmonic_potential(ax, x, omega)
        _combined_legend(ax, right, ncol=3)
    axes[-1].set_xlabel("x")
    fig.suptitle(f"{method.upper()} " + ("orbital probability densities" if squared else "molecular orbitals"))
    _save(fig, path)


def plot_orbitals_with_potential(x, up, down, omega, n_up, n_down, method, path):
    _plot_orbital_quantity(x, up, down, omega, n_up, n_down, method, path, squared=False)


def plot_orbital_densities_with_potential(x, up, down, omega, n_up, n_down, method, path):
    _plot_orbital_quantity(x, up, down, omega, n_up, n_down, method, path, squared=True)


def plot_pairs(pair, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); r = pair["r"]
    for key, lab in [("p_uu", r"$\uparrow\uparrow$"), ("p_dd", r"$\downarrow\downarrow$"), ("p_ud", r"$\uparrow\downarrow$")]:
        if key in pair: ax.plot(r, pair[key], label=lab)
    ax.set_xlabel(r"pair distance $r$"); ax.set_ylabel("normalized pair-distance density")
    ax.legend(frameon=False); _save(fig, path)


def plot_conditional(cond, classical_sites, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); x = cond["x"]
    ax.plot(x, cond["rho_cond_total"], label="total")
    ax.plot(x, cond["rho_cond_up"], ls="--", label="up")
    ax.plot(x, cond["rho_cond_down"], ls=":", label="down")
    ax.axvline(cond["pin_x"], lw=1.0, alpha=0.7)
    for site in np.asarray(classical_sites): ax.axvline(site, lw=0.6, alpha=0.25)
    ax.set_xlabel("x"); ax.set_ylabel(r"conditional density $\rho(x|x_0)$"); ax.set_ylim(ymin=0)
    ax.legend(frameon=False); _save(fig, path)


def plot_energy_trace(energy, running, path):
    e = np.asarray(energy)
    trace = e.mean(axis=1) if e.ndim == 2 else e
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(np.arange(1, len(trace) + 1), trace, alpha=0.45, label="sweep mean")
    ax.plot(np.arange(1, len(running) + 1), running, lw=1.5, label="running mean")
    ax.set_xlabel("production sweep"); ax.set_ylabel("local energy")
    ax.legend(frameon=False); _save(fig, path)


def plot_acf(acf, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); ax.plot(np.arange(len(acf)), acf)
    ax.axhline(0.0, lw=0.7); ax.set_xlabel("lag"); ax.set_ylabel("ACF")
    _save(fig, path)


def plot_block_curve(rows, path):
    if not rows: return
    a = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); ax.plot(a[:, 0], a[:, 2], marker="o", ms=3)
    ax.set_xscale("log"); ax.set_xlabel("block size (sweeps)"); ax.set_ylabel("block standard error")
    _save(fig, path)


def plot_natural_occupations(occ_up, occ_down, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(np.arange(1, len(occ_up) + 1), occ_up, marker="o", label="up")
    ax.plot(np.arange(1, len(occ_down) + 1), occ_down, marker="s", label="down")
    ax.set_xlabel("natural orbital index"); ax.set_ylabel("occupation")
    ax.set_ylim(bottom=0); ax.legend(frameon=False); _save(fig, path)
