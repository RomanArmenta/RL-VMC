"""JAX/Flax + e3nn-jax equivariant actor with SchNet-style radial filters.

Input: nuclei_xyz (Nn,3), electrons_xyz (Ne,3), nuclear_charges (Nn,)
Output for electron i: mu_i (3,), Sigma_i (3,3)

Architecture:
  e-N and e-e relative geometry
  -> Gaussian RBF -> SchNet MLP -> learned envelope c(r)
  -> spherical harmonics -> tensor products -> equivariant messages
  -> aggregation -> hidden irreps
  -> independent mu (1o) and covariance (0e+2e) heads
  -> Sigma = expm(A) + sigma_floor^2 I

Dependencies:
  pip install flax e3nn-jax matplotlib
  plus the appropriate JAX installation for your CPU/GPU.
"""

from __future__ import annotations
from typing import Literal, Optional
import math
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import linen as nn
import e3nn_jax as e3nn

Array = jax.Array


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(float(x)))


class LearnedEnvelope(nn.Module):
    initial_length: float
    initial_power: float = 2.0
    min_length: float = 1e-3
    min_power: float = 1.0

    @nn.compact
    def __call__(self, r, return_parameters=False):
        raw_l = self.param(
            "raw_length",
            lambda k: jnp.asarray(_inv_softplus(self.initial_length - self.min_length)),
        )
        raw_p = self.param(
            "raw_power",
            lambda k: jnp.asarray(_inv_softplus(self.initial_power - self.min_power)),
        )
        ell = jax.nn.softplus(raw_l) + self.min_length
        p = jax.nn.softplus(raw_p) + self.min_power
        c = jnp.exp(-jnp.maximum(r / ell, 0.0) ** p)
        if return_parameters:
            return c, ell, p
        return c


def gaussian_rbf(r, n_rbf=32, r_max=8.0):
    r = jnp.asarray(r)
    if r.ndim == 1:
        r = r[:, None]
    centers = jnp.linspace(0.0, r_max, n_rbf)
    spacing = r_max / (n_rbf - 1)
    gamma = 1.0 / spacing**2
    return jnp.exp(-gamma * (r - centers) ** 2)


class SchNetFilter(nn.Module):
    n_rbf: int = 32
    r_max: float = 8.0
    hidden: int = 64
    channels: int = 16

    @nn.compact
    def __call__(self, r):
        basis = gaussian_rbf(r, self.n_rbf, self.r_max)
        x = nn.Dense(self.hidden, name="dense0")(basis)
        x = jax.nn.softplus(x) - math.log(2.0)
        x = nn.Dense(self.hidden, name="dense1")(x)
        x = jax.nn.softplus(x) - math.log(2.0)
        raw = nn.Dense(self.channels, name="dense_out")(x)
        return basis, raw


class E3NNJAXActor(nn.Module):
    lmax: int = 2
    n_rbf: int = 32
    rbf_max: float = 8.0
    filter_hidden: int = 64
    radial_channels: int = 16
    eN_length: float = 2.0
    eN_power: float = 2.0
    ee_length: float = 2.5
    ee_power: float = 2.0
    irreps_hidden: str = "12x0e + 8x1o + 4x2e"
    sigma_floor: float = 0.05
    log_cov_scale: float = 0.25
    normalize_messages: bool = True

    def setup(self):
        self.env_eN = LearnedEnvelope(self.eN_length, self.eN_power, name="env_eN")
        self.env_ee = LearnedEnvelope(self.ee_length, self.ee_power, name="env_ee")
        self.filter_eN = SchNetFilter(
            self.n_rbf, self.rbf_max, self.filter_hidden, self.radial_channels,
            name="filter_eN",
        )
        self.filter_ee = SchNetFilter(
            self.n_rbf, self.rbf_max, self.filter_hidden, self.radial_channels,
            name="filter_ee",
        )
        self.msg_linear_eN = e3nn.flax.Linear(self.irreps_hidden, name="msg_linear_eN")
        self.msg_linear_ee = e3nn.flax.Linear(self.irreps_hidden, name="msg_linear_ee")
        self.hidden_linear = e3nn.flax.Linear(self.irreps_hidden, name="hidden_linear")
        self.mu_head = e3nn.flax.Linear("1x1o", name="mu_head")
        self.cov_head = e3nn.flax.Linear("1x0e + 1x2e", name="cov_head")

    @staticmethod
    def relative_geometry(points, center, eps=1e-12):
        rel = points - center
        dist = jnp.maximum(jnp.linalg.norm(rel, axis=-1, keepdims=True), eps)
        return rel, dist

    def _sh(self, rel):
        return e3nn.spherical_harmonics(
            list(range(self.lmax + 1)), rel,
            normalize=True, normalization="component",
        )

    def _radial_angular(self, rel, dist, channel: Literal["eN", "ee"]):
        if channel == "eN":
            _, raw = self.filter_eN(dist)
            c = self.env_eN(dist)
        else:
            _, raw = self.filter_ee(dist)
            c = self.env_ee(dist)
        effective = raw * c
        radial = e3nn.IrrepsArray(f"{self.radial_channels}x0e", effective)
        return e3nn.tensor_product(radial, self._sh(rel), regroup_output=True)

    def _messages_eN(self, rel, dist, Z):
        ar = self._radial_angular(rel, dist, "eN")
        src = e3nn.IrrepsArray(
            "2x0e",
            jnp.stack([jnp.ones_like(dist[:, 0]), Z / 10.0], axis=-1),
        )
        x = e3nn.tensor_product(src, ar, regroup_output=True)
        return self.msg_linear_eN(x)

    def _messages_ee(self, rel, dist):
        ar = self._radial_angular(rel, dist, "ee")
        src = e3nn.IrrepsArray("1x0e", jnp.ones((rel.shape[0], 1), dtype=rel.dtype))
        x = e3nn.tensor_product(src, ar, regroup_output=True)
        return self.msg_linear_ee(x)

    def encode_environment(self, nuclei, electrons, electron_index: int, Z):
        ri = electrons[electron_index]

        rel_n, dist_n = self.relative_geometry(nuclei, ri)
        h_n = e3nn.sum(self._messages_eN(rel_n, dist_n, Z), axis=0)
        if self.normalize_messages:
            h_n = h_n / math.sqrt(max(1, nuclei.shape[0]))

        # electron_index is intended to be a static Python int under jit
        others = jnp.concatenate(
            [electrons[:electron_index], electrons[electron_index + 1:]], axis=0
        )
        if others.shape[0] == 0:
            h_e = e3nn.zeros(self.irreps_hidden, dtype=electrons.dtype)
        else:
            rel_e, dist_e = self.relative_geometry(others, ri)
            h_e = e3nn.sum(self._messages_ee(rel_e, dist_e), axis=0)
            if self.normalize_messages:
                h_e = h_e / math.sqrt(max(1, others.shape[0]))

        return self.hidden_linear(h_n + h_e)

    @staticmethod
    def symmetric_irreps_to_cartesian(coeffs):
        # Exact basis for a symmetric product of two 1o vectors: 0e + 2e.
        basis = e3nn.reduced_symmetric_tensor_product_basis(
            e3nn.Irreps("1o"), degree=2
        )
        coeffs = coeffs.rechunk(basis.irreps)
        A = jnp.einsum("k,ijk->ij", coeffs.array, basis.array)
        return 0.5 * (A + A.T)

    def __call__(
        self,
        nuclei_xyz,
        electrons_xyz,
        electron_index: int = 0,
        nuclear_charges: Optional[Array] = None,
        return_log_covariance: bool = False,
    ):
        nuclei = jnp.asarray(nuclei_xyz)
        electrons = jnp.asarray(electrons_xyz)
        Z = (
            jnp.ones((nuclei.shape[0],), dtype=nuclei.dtype)
            if nuclear_charges is None
            else jnp.asarray(nuclear_charges, dtype=nuclei.dtype)
        )

        h = self.encode_environment(nuclei, electrons, electron_index, Z)
        mu = self.mu_head(h).array

        A_ir = self.cov_head(h)
        A = self.log_cov_scale * self.symmetric_irreps_to_cartesian(A_ir)
        Sigma = jsp.linalg.expm(A) + self.sigma_floor**2 * jnp.eye(3, dtype=A.dtype)

        if return_log_covariance:
            return mu, Sigma, A
        return mu, Sigma

    def radial_diagnostics(self, r, channel: Literal["eN", "ee"]):
        r = jnp.asarray(r)
        if r.ndim == 1:
            r = r[:, None]
        if channel == "eN":
            basis, raw = self.filter_eN(r)
            c, ell, p = self.env_eN(r, return_parameters=True)
        else:
            basis, raw = self.filter_ee(r)
            c, ell, p = self.env_ee(r, return_parameters=True)
        return {
            "basis": basis,
            "raw": raw,
            "envelope": c,
            "effective": raw * c,
            "length": ell,
            "power": p,
        }


def rotation_matrix(axis=(1.0, 0.7, -0.25), angle=0.91, dtype=jnp.float64):
    axis = jnp.asarray(axis, dtype=dtype)
    axis = axis / jnp.linalg.norm(axis)
    x, y, z = axis
    K = jnp.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=dtype)
    I = jnp.eye(3, dtype=dtype)
    return I + jnp.sin(angle) * K + (1.0 - jnp.cos(angle)) * (K @ K)


def test_equivariance(model, variables, nuclei, electrons, Z, electron_index=0):
    mu, Sigma, A = model.apply(
        variables, nuclei, electrons, electron_index, Z, True
    )
    R = rotation_matrix(dtype=nuclei.dtype)
    nr = nuclei @ R.T
    er = electrons @ R.T
    mu_r, Sigma_r, A_r = model.apply(
        variables, nr, er, electron_index, Z, True
    )
    return {
        "mu_error": jnp.linalg.norm(mu_r - R @ mu),
        "A_error": jnp.linalg.norm(A_r - R @ A @ R.T),
        "Sigma_error": jnp.linalg.norm(Sigma_r - R @ Sigma @ R.T),
    }


def count_parameters(variables) -> int:
    params = variables["params"] if "params" in variables else variables
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def plot_envelopes(model, variables, r_max=8.0, n_points=500):
    import matplotlib.pyplot as plt
    r = jnp.linspace(0.0, r_max, n_points)
    eN = model.apply(variables, r, "eN", method=model.radial_diagnostics)
    ee = model.apply(variables, r, "ee", method=model.radial_diagnostics)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        jax.device_get(r), jax.device_get(eN["envelope"]).squeeze(),
        label=f"c_eN: ell={float(eN['length']):.3f}, p={float(eN['power']):.3f}",
    )
    ax.plot(
        jax.device_get(r), jax.device_get(ee["envelope"]).squeeze(), "--",
        label=f"c_ee: ell={float(ee['length']):.3f}, p={float(ee['power']):.3f}",
    )
    ax.set_xlabel("distance r")
    ax.set_ylabel("c(r)")
    ax.set_title("Learned e-N and e-e envelopes")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_rbf_basis(model, variables, channel="eN", r_max=8.0, n_points=500, max_basis=None):
    import matplotlib.pyplot as plt
    r = jnp.linspace(0.0, r_max, n_points)
    out = model.apply(variables, r, channel, method=model.radial_diagnostics)
    basis = jax.device_get(out["basis"])
    n = basis.shape[-1] if max_basis is None else min(max_basis, basis.shape[-1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in range(n):
        ax.plot(jax.device_get(r), basis[:, k], lw=1.0, alpha=0.75)
    ax.set_xlabel("distance r")
    ax.set_ylabel("phi_k(r)")
    ax.set_title(f"Gaussian radial basis — {channel}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_radial_filters(
    model, variables, channel="eN", r_max=8.0, n_points=500,
    n_channels=8, effective=True,
):
    import matplotlib.pyplot as plt
    r = jnp.linspace(0.0, r_max, n_points)
    out = model.apply(variables, r, channel, method=model.radial_diagnostics)
    values = out["effective"] if effective else out["raw"]
    values = jax.device_get(values)
    n = min(n_channels, values.shape[-1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in range(n):
        ax.plot(jax.device_get(r), values[:, k], label=f"filter {k}")
    ax.set_xlabel("distance r")
    ax.set_ylabel("c(r) w_k(r)" if effective else "w_k(r)")
    ax.set_title(("Effective" if effective else "Raw") + f" radial filters — {channel}")
    if n <= 10:
        ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def demo():
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(7)

    nuclei = jnp.array([
        [0.000, 0.000, 0.000],
        [0.958, 0.000, 0.000],
        [-0.240, 0.927, 0.000],
    ], dtype=jnp.float64)
    Z = jnp.array([8.0, 1.0, 1.0], dtype=jnp.float64)
    electrons = jnp.array([
        [ 0.35,  0.25,  0.15],
        [-0.45,  0.30, -0.20],
        [ 0.20, -0.55,  0.35],
        [-0.10,  0.15,  0.80],
        [ 0.65,  0.05, -0.45],
        [-0.55, -0.35,  0.10],
    ], dtype=jnp.float64)

    model = E3NNJAXActor()
    variables = model.init(key, nuclei, electrons, 0, Z, True)

    mu, Sigma, A = model.apply(variables, nuclei, electrons, 0, Z, True)
    print("mu =\n", mu)
    print("A =\n", A)
    print("Sigma =\n", Sigma)
    print("eig(Sigma) =\n", jnp.linalg.eigvalsh(Sigma))
    print("parameters =", count_parameters(variables))
    print("equivariance =", test_equivariance(model, variables, nuclei, electrons, Z))

    plot_envelopes(model, variables)
    plot_rbf_basis(model, variables, "eN", max_basis=16)
    plot_radial_filters(model, variables, "eN", effective=False)
    plot_radial_filters(model, variables, "eN", effective=True)
    plot_radial_filters(model, variables, "ee", effective=True)

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    demo()