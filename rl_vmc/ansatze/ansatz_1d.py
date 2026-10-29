from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import re
from typing import Literal
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from rl_vmc.config import Experiment1DConfig

Array = jax.Array
OrbitalModel = Literal["ho_fixed", "ho_variational", "gaussian_basis", "backflow_ho"]
CorrelationModel = Literal[
    "none",
    "jastrow",
    "spin_rbf",
    "spin_rbf_jastrow",
    "spin_rbf_threebody",
]


def _softplus_inverse(y: float) -> float:
    if y <= 0:
        raise ValueError("softplus inverse requires y > 0")
    return math.log(math.expm1(float(y)))


@dataclass(frozen=True)
class _System1DSpec:
    n_electrons: int = 4
    n_up: int = 2
    omega: float = 1.0
    interaction_strength: float = 1.0
    softening: float = 0.25

    @property
    def n_down(self) -> int:
        return self.n_electrons - self.n_up

    @property
    def max_spin_occupancy(self) -> int:
        return max(self.n_up, self.n_down)


@dataclass(frozen=True)
class _Orbital1DSpec:
    model: OrbitalModel = "ho_fixed"
    alpha_init: float = 1.0
    min_positive: float = 1e-5
    gaussian_basis_size: int = 10
    gaussian_center_max: float = 4.0
    gaussian_width_init: float = 0.9
    gaussian_train_width: bool = True
    backflow_components: int = 8
    backflow_r_max: float = 5.0
    backflow_width: float = 0.7
    backflow_envelope_length: float = 3.0


@dataclass(frozen=True)
class _Correlation1DSpec:
    model: CorrelationModel = "none"
    min_positive: float = 1e-6
    pair_smoothing: float = 1e-4
    jastrow_c_same: float = 0.15
    jastrow_c_opposite: float = 0.30
    jastrow_beta_same: float = 0.6
    jastrow_beta_opposite: float = 0.5
    rbf_components: int = 10
    rbf_r_max: float = 5.0
    rbf_width_init: float = 0.6
    rbf_coeff_init: float = 0.08
    threebody_components: int = 6
    threebody_q_max: float = 8.0
    threebody_width_init: float = 1.0
    threebody_coeff_init: float = 0.02


def _spin_labels(cfg: _System1DSpec) -> Array:
    return jnp.concatenate(
        [
            jnp.ones((cfg.n_up,), dtype=jnp.int32),
            -jnp.ones((cfg.n_down,), dtype=jnp.int32),
        ]
    )


def _pair_indices(n_electrons: int) -> tuple[Array, Array]:
    pairs = list(itertools.combinations(range(int(n_electrons)), 2))
    if not pairs:
        return jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,), dtype=jnp.int32)
    i, j = zip(*pairs)
    return jnp.asarray(i, dtype=jnp.int32), jnp.asarray(j, dtype=jnp.int32)


def _pair_distances(x: Array, cfg: _System1DSpec) -> Array:
    i, j = _pair_indices(cfg.n_electrons)
    if i.size == 0:
        return jnp.zeros(x.shape[:-1] + (0,), dtype=x.dtype)
    return jnp.abs(x[..., i] - x[..., j])


def _pair_spin_masks(cfg: _System1DSpec) -> tuple[Array, Array]:
    i, j = _pair_indices(cfg.n_electrons)
    s = _spin_labels(cfg)
    same = s[i] == s[j]
    return same, ~same


def _hermite_physicists(n: int, z: Array) -> Array:
    if n == 0:
        return jnp.ones_like(z)
    if n == 1:
        return 2.0 * z
    hm2 = jnp.ones_like(z)
    hm1 = 2.0 * z
    for k in range(1, n):
        h = 2.0 * z * hm1 - 2.0 * k * hm2
        hm2, hm1 = hm1, h
    return hm1


def _harmonic_orbital(n: int, x: Array, alpha: Array) -> Array:
    z = jnp.sqrt(alpha) * x
    norm = (alpha / jnp.pi) ** 0.25 / math.sqrt((2.0**n) * math.factorial(n))
    return norm * _hermite_physicists(n, z) * jnp.exp(-0.5 * alpha * x**2)


def _harmonic_orbital_bank(x: Array, n_orbitals: int, alpha: Array) -> Array:
    return jnp.stack([_harmonic_orbital(n, x, alpha) for n in range(n_orbitals)], axis=-1)


def _gaussian_centers(cfg: _Orbital1DSpec) -> Array:
    return jnp.linspace(-cfg.gaussian_center_max, cfg.gaussian_center_max, cfg.gaussian_basis_size)


def _gaussian_basis_values(x: Array, centers: Array, width: Array) -> Array:
    return jnp.exp(-0.5 * ((x[..., None] - centers) / width) ** 2)


def _initial_gaussian_coefficients(system_cfg: _System1DSpec, cfg: _Orbital1DSpec) -> Array:
    n_orb = system_cfg.max_spin_occupancy
    K = cfg.gaussian_basis_size
    if K < n_orb:
        raise ValueError(f"gaussian_basis_size ({K}) must be >= max spin occupancy ({n_orb})")
    grid = np.linspace(-cfg.gaussian_center_max * 1.3, cfg.gaussian_center_max * 1.3, 500)
    centers = np.linspace(-cfg.gaussian_center_max, cfg.gaussian_center_max, K)
    width = float(cfg.gaussian_width_init)
    B = np.exp(-0.5 * ((grid[:, None] - centers[None, :]) / width) ** 2)
    alpha = float(cfg.alpha_init)
    coeffs = []
    z = np.sqrt(alpha) * grid
    H = [np.ones_like(z)]
    if n_orb > 1:
        H.append(2.0 * z)
    for n in range(1, n_orb - 1):
        H.append(2.0 * z * H[-1] - 2.0 * n * H[-2])
    for n in range(n_orb):
        norm = (alpha / np.pi) ** 0.25 / math.sqrt((2.0**n) * math.factorial(n))
        target = norm * H[n] * np.exp(-0.5 * alpha * grid**2)
        c, *_ = np.linalg.lstsq(B, target, rcond=1e-10)
        coeffs.append(c)
    return jnp.asarray(np.stack(coeffs, axis=0))


def init_orbital_params(key: Array, system_cfg: _System1DSpec, cfg: _Orbital1DSpec) -> dict:
    eps = cfg.min_positive
    if cfg.model == "ho_fixed":
        return {}
    if cfg.model == "ho_variational":
        return {"raw_orbital_alpha": jnp.asarray(_softplus_inverse(max(cfg.alpha_init - eps, eps)))}
    if cfg.model == "gaussian_basis":
        params = {"orbital_coeff": _initial_gaussian_coefficients(system_cfg, cfg)}
        if cfg.gaussian_train_width:
            params["raw_gaussian_width"] = jnp.asarray(
                _softplus_inverse(max(cfg.gaussian_width_init - eps, eps))
            )
        return params
    if cfg.model == "backflow_ho":
        return {
            "raw_orbital_alpha": jnp.asarray(_softplus_inverse(max(cfg.alpha_init - eps, eps))),
            "backflow_coeff_same": jnp.zeros((cfg.backflow_components,)),
            "backflow_coeff_opposite": jnp.zeros((cfg.backflow_components,)),
        }
    raise ValueError(f"Unknown orbital model: {cfg.model}")


def _infer_orbital_model_from_params(orb_params: dict, fallback: str) -> str:
    if not isinstance(orb_params, dict):
        return fallback
    if "backflow_coeff_same" in orb_params:
        return "backflow_ho"
    if "orbital_coeff" in orb_params:
        return "gaussian_basis"
    if "raw_orbital_alpha" in orb_params:
        return "ho_variational"
    if len(orb_params) == 0:
        return "ho_fixed"
    return fallback


def _physical_orbital_params(params: dict, cfg: _Orbital1DSpec) -> tuple[dict, str]:
    eps = cfg.min_positive
    eff_model = _infer_orbital_model_from_params(params, cfg.model)
    if eff_model == "ho_fixed":
        return {"alpha": jnp.asarray(cfg.alpha_init)}, eff_model
    if eff_model == "ho_variational":
        return {"alpha": jax.nn.softplus(params["raw_orbital_alpha"]) + eps}, eff_model
    if eff_model == "gaussian_basis":
        width = (
            jax.nn.softplus(params["raw_gaussian_width"]) + eps
            if "raw_gaussian_width" in params
            else jnp.asarray(cfg.gaussian_width_init)
        )
        coeff = params["orbital_coeff"]
        coeff = coeff / jnp.sqrt(jnp.sum(coeff**2, axis=1, keepdims=True) + 1e-12)
        return {"coeff": coeff, "centers": _gaussian_centers(cfg), "width": width}, eff_model
    if eff_model == "backflow_ho":
        return {
            "alpha": jax.nn.softplus(params["raw_orbital_alpha"]) + eps,
            "backflow_coeff_same": params["backflow_coeff_same"],
            "backflow_coeff_opposite": params["backflow_coeff_opposite"],
        }, eff_model
    raise ValueError(f"Unknown orbital model: {eff_model}")


def _backflow_displacement(x: Array, system_cfg: _System1DSpec, cfg: _Orbital1DSpec, p: dict) -> Array:
    N = system_cfg.n_electrons
    if N <= 1:
        return jnp.zeros_like(x)
    s = _spin_labels(system_cfg)
    n_comp = int(p["backflow_coeff_same"].shape[0])
    centers = jnp.linspace(0.0, cfg.backflow_r_max, n_comp)
    width = jnp.asarray(cfg.backflow_width, dtype=x.dtype)
    ell = jnp.asarray(cfg.backflow_envelope_length, dtype=x.dtype)
    dx = x[:, None] - x[None, :]
    r = jnp.abs(dx)
    basis = jnp.exp(-0.5 * ((r[..., None] - centers) / width) ** 2)
    envelope = jnp.exp(-(r / ell) ** 2)
    eta_same = jnp.einsum("ijk,k->ij", basis, p["backflow_coeff_same"])
    eta_opp = jnp.einsum("ijk,k->ij", basis, p["backflow_coeff_opposite"])
    same = s[:, None] == s[None, :]
    eta = jnp.where(same, eta_same, eta_opp) * envelope
    mask = 1.0 - jnp.eye(N, dtype=x.dtype)
    return jnp.sum(mask * dx * eta, axis=1) / math.sqrt(max(N - 1, 1))


def determinant_logabs_sign(
    x: Array,
    params: dict,
    system_cfg: _System1DSpec,
    cfg: _Orbital1DSpec,
    node_epsilon: float = 1e-14,
) -> tuple[Array, Array]:
    n_orb = system_cfg.max_spin_occupancy
    if n_orb == 0:
        return jnp.asarray(0.0, dtype=x.dtype), jnp.asarray(1.0, dtype=x.dtype)

    p, eff_model = _physical_orbital_params(params, cfg)
    xeff = x + _backflow_displacement(x, system_cfg, cfg, p) if eff_model == "backflow_ho" else x

    if eff_model in ("ho_fixed", "ho_variational", "backflow_ho"):
        bank = _harmonic_orbital_bank(xeff, n_orb, p["alpha"])
    elif eff_model == "gaussian_basis":
        B = _gaussian_basis_values(xeff, p["centers"], p["width"])
        bank = B @ p["coeff"].T
    else:
        raise ValueError(f"Unknown orbital model: {eff_model}")

    total_logabs = jnp.asarray(0.0, dtype=x.dtype)
    total_sign = jnp.asarray(1.0, dtype=x.dtype)
    for start, nspin in ((0, system_cfg.n_up), (system_cfg.n_up, system_cfg.n_down)):
        if nspin == 0:
            continue
        M = bank[start : start + nspin, :nspin]
        sign, logabs = jnp.linalg.slogdet(M)
        logabs = jnp.where(jnp.isfinite(logabs), logabs, jnp.log(node_epsilon))
        total_logabs = total_logabs + logabs
        total_sign = total_sign * sign
    return total_logabs, total_sign


def init_correlation_params(key: Array, cfg: _Correlation1DSpec) -> dict:
    eps = cfg.min_positive
    if cfg.model == "none":
        return {}
    params: dict[str, Array] = {}
    if cfg.model in ("jastrow", "spin_rbf_jastrow"):
        params.update(
            {
                "raw_c_same": jnp.asarray(_softplus_inverse(max(cfg.jastrow_c_same - eps, eps))),
                "raw_c_opposite": jnp.asarray(_softplus_inverse(max(cfg.jastrow_c_opposite - eps, eps))),
                "raw_beta_same": jnp.asarray(_softplus_inverse(max(cfg.jastrow_beta_same - eps, eps))),
                "raw_beta_opposite": jnp.asarray(_softplus_inverse(max(cfg.jastrow_beta_opposite - eps, eps))),
            }
        )
    if cfg.model in ("spin_rbf", "spin_rbf_jastrow", "spin_rbf_threebody"):
        K = cfg.rbf_components
        raw_a = _softplus_inverse(max(cfg.rbf_coeff_init - eps, eps))
        params.update(
            {
                "raw_pair_coeff_same": jnp.full((K,), raw_a),
                "raw_pair_coeff_opposite": jnp.full((K,), raw_a),
                "raw_pair_width_same": jnp.asarray(_softplus_inverse(max(cfg.rbf_width_init - eps, eps))),
                "raw_pair_width_opposite": jnp.asarray(_softplus_inverse(max(cfg.rbf_width_init - eps, eps))),
            }
        )
    if cfg.model == "spin_rbf_threebody":
        K3 = cfg.threebody_components
        raw_a3 = _softplus_inverse(max(cfg.threebody_coeff_init - eps, eps))
        params.update(
            {
                "raw_threebody_coeff": jnp.full((K3,), raw_a3),
                "raw_threebody_width": jnp.asarray(_softplus_inverse(max(cfg.threebody_width_init - eps, eps))),
            }
        )
    return params


def _infer_correlation_model_from_params(corr_params: dict, fallback: str) -> str:
    if not isinstance(corr_params, dict):
        return fallback
    has_j = "raw_c_same" in corr_params
    has_rbf = "raw_pair_coeff_same" in corr_params
    has_3b = "raw_threebody_coeff" in corr_params
    if has_3b:
        return "spin_rbf_threebody"
    if has_j and has_rbf:
        return "spin_rbf_jastrow"
    if has_rbf:
        return "spin_rbf"
    if has_j:
        return "jastrow"
    if len(corr_params) == 0:
        return "none"
    return fallback


def log_correlation_factor(
    params: dict,
    x: Array,
    system_cfg: _System1DSpec,
    cfg: _Correlation1DSpec,
) -> Array:
    eff_model = _infer_correlation_model_from_params(params, cfg.model)
    if eff_model == "none":
        return jnp.asarray(0.0, dtype=x.dtype)

    eps = cfg.min_positive
    r = _pair_distances(x, system_cfg)
    same, _ = _pair_spin_masks(system_cfg)
    total = jnp.asarray(0.0, dtype=x.dtype)

    if eff_model in ("jastrow", "spin_rbf_jastrow"):
        c_same = jax.nn.softplus(params["raw_c_same"]) + eps
        c_opp = jax.nn.softplus(params["raw_c_opposite"]) + eps
        b_same = jax.nn.softplus(params["raw_beta_same"]) + eps
        b_opp = jax.nn.softplus(params["raw_beta_opposite"]) + eps
        rho = jnp.sqrt(r**2 + cfg.pair_smoothing**2)
        u_same = c_same * rho / (1.0 + b_same * rho)
        u_opp = c_opp * rho / (1.0 + b_opp * rho)
        total = total + jnp.sum(jnp.where(same, u_same, u_opp), axis=-1)

    if eff_model in ("spin_rbf", "spin_rbf_jastrow", "spin_rbf_threebody"):
        pc_same = jax.nn.softplus(params["raw_pair_coeff_same"]) + eps
        pc_opp = jax.nn.softplus(params["raw_pair_coeff_opposite"]) + eps
        pw_same = jax.nn.softplus(params["raw_pair_width_same"]) + eps
        pw_opp = jax.nn.softplus(params["raw_pair_width_opposite"]) + eps
        n_rbf = int(pc_same.shape[0])
        centers = jnp.linspace(0.0, cfg.rbf_r_max, n_rbf)
        r_exp = jnp.asarray(r)[..., None]
        gs = jnp.exp(-0.5 * ((r_exp - centers) / pw_same) ** 2)
        go = jnp.exp(-0.5 * ((r_exp - centers) / pw_opp) ** 2)
        fs = 1.0 + jnp.sum(pc_same * gs, axis=-1)
        fo = 1.0 + jnp.sum(pc_opp * go, axis=-1)
        total = total + jnp.sum(jnp.log(jnp.where(same, fs, fo)), axis=-1)

    if eff_model == "spin_rbf_threebody" and system_cfg.n_electrons >= 3:
        tb_coeff = jax.nn.softplus(params["raw_threebody_coeff"]) + eps
        tb_width = jax.nn.softplus(params["raw_threebody_width"]) + eps
        n_3b = int(tb_coeff.shape[0])
        centers3 = jnp.linspace(0.0, cfg.threebody_q_max, n_3b)
        terms = []
        for i, j, k in itertools.combinations(range(system_cfg.n_electrons), 3):
            rij = jnp.abs(x[i] - x[j])
            rik = jnp.abs(x[i] - x[k])
            rjk = jnp.abs(x[j] - x[k])
            q = jnp.sqrt(rij**2 + rik**2 + rjk**2)
            basis = jnp.exp(-0.5 * ((q - centers3) / tb_width) ** 2)
            terms.append(jnp.log(1.0 + jnp.sum(tb_coeff * basis)))
        if terms:
            total = total + jnp.sum(jnp.stack(terms))

    return total


def _build_ho_basis_jax(r: jax.Array, n_basis: int, omega: float) -> jax.Array:
    """Evalúa las autofunciones del oscilador armónico 1D hasta n_basis-1 en las posiciones r."""
    z = jnp.sqrt(omega) * r
    H = [jnp.ones_like(z), 2.0 * z]
    for n in range(2, n_basis):
        H.append(2.0 * z * H[n - 1] - 2.0 * (n - 1) * H[n - 2])
    H_stack = jnp.stack(H[:n_basis], axis=0)
    ns = jnp.arange(n_basis)
    fact = jnp.exp(gammaln(ns + 1.0))
    norm = jnp.sqrt(2.0 ** ns * fact)
    pref = (omega / jnp.pi) ** 0.25 * jnp.exp(-0.5 * z ** 2)
    return (pref * (H_stack / norm[:, None])).T


class BaseAnsatz1D:
    """Interfaz base para funciones de onda 1D que retornan log|Psi(X)|."""

    trainable: bool = True

    def init(self, key: jax.Array, walkers: jax.Array) -> dict:
        raise NotImplementedError

    def apply(self, params: dict, walkers: jax.Array) -> jax.Array:
        raise NotImplementedError

    def value_and_grad(
        self,
        params: dict,
        walkers: jax.Array,
        min_clip: float = -500.0,
        max_clip: float = 500.0,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Evalúa log|Psi(X)| y grad_X log|Psi(X)| para un lote de caminantes.
        walkers: shape (batch, n_elec, 1)
        Retorna: (log_psi, grad_log_psi) con shapes (batch,) y (batch, n_elec, 1)
        """
        def single_log_psi(w_single: jax.Array) -> jax.Array:
            val = self.apply(params, w_single[None, ...])
            return jnp.squeeze(val)

        vg_fn = jax.vmap(jax.value_and_grad(single_log_psi))
        log_psi, grad_log_psi = vg_fn(walkers)
        grad_log_psi = jnp.nan_to_num(
            grad_log_psi, nan=0.0, posinf=max_clip, neginf=min_clip
        )
        grad_log_psi = jnp.clip(grad_log_psi, min_clip, max_clip)
        return log_psi, grad_log_psi


class SlaterJastrow1DAnsatz(BaseAnsatz1D):
    """
    Ansatz 1D modular Slater-Jastrow-Backflow completamente independiente de `src/`.
    Mantiene 100% compatibilidad de estructura de pesos PyTree (`{"orbitals": ..., "correlation": ...}`)
    con los archivos `.pkl` generados por `src/wavefunction.py` y `Ansatz1D_Backflow`, e infiere
    automáticamente la variante de orbitales/correlación a partir de las llaves del diccionario cargado.
    """

    def __init__(self, cfg: Experiment1DConfig):
        self.cfg = cfg
        self.trainable = True
        self.node_epsilon = 1e-14
        self.sys_cfg = _System1DSpec(
            n_electrons=cfg.system.n_elec,
            n_up=cfg.system.n_up,
            omega=cfg.system.omega,
            interaction_strength=cfg.system.interaction_strength,
            softening=cfg.system.softening,
        )
        self.orbital_cfg = _Orbital1DSpec(
            model=cfg.ansatz.orbital_model,
            backflow_components=cfg.ansatz.backflow_components,
            backflow_r_max=cfg.ansatz.backflow_r_max,
            backflow_width=cfg.ansatz.backflow_width,
            backflow_envelope_length=cfg.ansatz.backflow_envelope_length,
        )
        self.correlation_cfg = _Correlation1DSpec(
            model=cfg.ansatz.correlation_model,
        )

    def init(self, key: jax.Array, walkers: jax.Array) -> dict:
        k1, k2 = jax.random.split(key)
        return {
            "orbitals": init_orbital_params(k1, self.sys_cfg, self.orbital_cfg),
            "correlation": init_correlation_params(k2, self.correlation_cfg),
        }

    def apply(self, params: dict, walkers: jax.Array) -> jax.Array:
        x_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        if x_1d.ndim == 1:
            x_1d = x_1d[None, :]

        def single_apply(x_single: jax.Array) -> jax.Array:
            logdet, _ = determinant_logabs_sign(
                x_single,
                params["orbitals"],
                self.sys_cfg,
                self.orbital_cfg,
                node_epsilon=self.node_epsilon,
            )
            logcorr = log_correlation_factor(
                params["correlation"],
                x_single,
                self.sys_cfg,
                self.correlation_cfg,
            )
            return logdet + logcorr

        return jax.vmap(single_apply)(x_1d)


class CASSCF1DAnsatz(BaseAnsatz1D):
    """
    Función de onda multideterminantal CASSCF 1D cargada desde un archivo `.npz`.
    Soporta evaluación congelada (`casscf_trainable=False`) u optimización VMC (`casscf_trainable=True`).
    """

    def __init__(self, cfg: Experiment1DConfig):
        self.cfg = cfg
        self.trainable = cfg.ansatz.casscf_trainable
        npz_path = cfg.ansatz.casscf_npz_path
        if not npz_path:
            raise ValueError("CASSCF1DAnsatz requiere especificar `--casscf_npz_path`.")

        data = np.load(npz_path)
        meta = json.loads(str(data["metadata_json"]))
        m_cfg = meta["config"]
        self.omega = float(m_cfg.get("omega", cfg.system.omega))
        self.n_up = int(m_cfg.get("n_up", cfg.system.n_up))
        self.n_dn = int(m_cfg.get("n_down", cfg.system.n_dn))
        self.n_basis = int(m_cfg.get("n_basis", 8))

        match = re.search(r"c(\d+)-a(\d+)", npz_path)
        fallback_core = int(match.group(1)) if match else 0
        fallback_active = int(match.group(2)) if match else 2
        self.n_core = int(m_cfg.get("n_core", m_cfg.get("n_core_orbitals", fallback_core)))
        self.n_active = int(
            m_cfg.get("n_active_orbitals", m_cfg.get("n_active", fallback_active))
        )

        self.orbitals_init = jnp.array(data["orbital_coefficients"])
        self.ci_coeffs_init = jnp.array(data["selected_ci_coefficients"])
        dets = data["selected_ci_determinants"]

        occ_up_list, occ_dn_list = [], []
        for d_str in dets:
            val = int(d_str)
            act_up = [i for i in range(self.n_active) if val & (1 << i)]
            act_dn = [
                i for i in range(self.n_active) if val & (1 << (self.n_active + i))
            ]
            occ_up_list.append(
                list(range(self.n_core)) + [self.n_core + i for i in act_up]
            )
            occ_dn_list.append(
                list(range(self.n_core)) + [self.n_core + i for i in act_dn]
            )

        self.occ_up_idx = jnp.array(occ_up_list)
        self.occ_dn_idx = jnp.array(occ_dn_list)

    def init(self, key: jax.Array, walkers: jax.Array) -> dict:
        return {
            "orbitals": self.orbitals_init,
            "ci_coeffs": self.ci_coeffs_init,
        }

    def apply(self, params: dict, walkers: jax.Array) -> jax.Array:
        orbitals = params.get("orbitals", self.orbitals_init)
        ci_coeffs = params.get("ci_coeffs", self.ci_coeffs_init)
        x_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        if x_1d.ndim == 1:
            x_1d = x_1d[None, :]

        n_up = self.n_up
        n_dn = self.n_dn
        n_basis = self.n_basis
        omega = self.omega
        occ_up_idx = self.occ_up_idx
        occ_dn_idx = self.occ_dn_idx

        def evaluate_single(x_single: jax.Array) -> jax.Array:
            x_up, x_dn = x_single[:n_up], x_single[n_up:]
            basis_up = _build_ho_basis_jax(x_up, n_basis, omega)
            basis_dn = _build_ho_basis_jax(x_dn, n_basis, omega)

            def eval_det(idx_up, idx_dn, c_i):
                mat_up = basis_up @ orbitals[:, idx_up] + jnp.eye(n_up) * 1e-12
                mat_dn = basis_dn @ orbitals[:, idx_dn] + jnp.eye(n_dn) * 1e-12
                s_u, l_u = jnp.linalg.slogdet(mat_up)
                s_d, l_d = jnp.linalg.slogdet(mat_dn)
                return c_i * s_u * s_d, l_u + l_d

            c_vals, log_mags = jax.vmap(eval_det)(occ_up_idx, occ_dn_idx, ci_coeffs)
            max_log = jnp.max(log_mags)
            psi_val = jnp.sum(c_vals * jnp.exp(log_mags - max_log))
            return jnp.clip(
                max_log + jnp.log(jnp.abs(psi_val) + 1e-15), -1000.0, 1000.0
            )

        return jax.vmap(evaluate_single)(x_1d)


def build_ansatz_1d(cfg: Experiment1DConfig) -> BaseAnsatz1D:
    if cfg.ansatz.ansatz_type in ("slater_jastrow_1d", "slater_jastrow"):
        return SlaterJastrow1DAnsatz(cfg)
    elif cfg.ansatz.ansatz_type in ("casscf_npz", "casscf_1d", "casscf"):
        return CASSCF1DAnsatz(cfg)
    else:
        raise ValueError(f"Tipo de ansatz no soportado: {cfg.ansatz.ansatz_type}")
