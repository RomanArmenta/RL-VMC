from __future__ import annotations
from dataclasses import dataclass
import itertools
import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class SystemConfig:
    """N interacting electrons in a 1D harmonic trap, atomic units.

    H = -1/2 sum_i d^2/dx_i^2
        + 1/2 omega^2 sum_i x_i^2
        + g sum_{i<j} 1/sqrt((x_i-x_j)^2 + softening^2)

    Spin is represented by fixed populations N_up and N_down. Electrons are
    indistinguishable within each spin sector; the Slater determinants enforce
    the corresponding antisymmetry.
    """

    n_electrons: int = 4
    n_up: int = 2
    omega: float = 1.0
    interaction_strength: float = 1.0
    softening: float = 0.25

    def __post_init__(self):
        if self.n_electrons < 1:
            raise ValueError("n_electrons must be >= 1")
        if not (0 <= self.n_up <= self.n_electrons):
            raise ValueError("n_up must satisfy 0 <= n_up <= n_electrons")
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        if self.softening <= 0:
            raise ValueError("softening must be positive")

    @property
    def n_down(self) -> int:
        return self.n_electrons - self.n_up

    @property
    def max_spin_occupancy(self) -> int:
        return max(self.n_up, self.n_down)


def spin_labels(cfg: SystemConfig) -> Array:
    """Return shape (N,), +1 for up and -1 for down."""
    return jnp.concatenate(
        [
            jnp.ones((cfg.n_up,), dtype=jnp.int32),
            -jnp.ones((cfg.n_down,), dtype=jnp.int32),
        ]
    )


def pair_indices(n_electrons: int) -> tuple[Array, Array]:
    pairs = list(itertools.combinations(range(int(n_electrons)), 2))
    if not pairs:
        return jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,), dtype=jnp.int32)
    i, j = zip(*pairs)
    return jnp.asarray(i, dtype=jnp.int32), jnp.asarray(j, dtype=jnp.int32)


def pair_distances(x: Array, cfg: SystemConfig) -> Array:
    """All |x_i-x_j|, shape (..., N_pairs)."""
    i, j = pair_indices(cfg.n_electrons)
    if i.size == 0:
        return jnp.zeros(x.shape[:-1] + (0,), dtype=x.dtype)
    return jnp.abs(x[..., i] - x[..., j])


def pair_spin_masks(cfg: SystemConfig) -> tuple[Array, Array]:
    """Boolean masks over the canonical i<j pair list: same, opposite spin."""
    i, j = pair_indices(cfg.n_electrons)
    s = spin_labels(cfg)
    same = s[i] == s[j]
    opposite = ~same
    return same, opposite


def harmonic_potential(x: Array, cfg: SystemConfig) -> Array:
    return 0.5 * (cfg.omega**2) * jnp.sum(jnp.asarray(x) ** 2, axis=-1)


def interaction_energy(x: Array, cfg: SystemConfig) -> Array:
    r = pair_distances(jnp.asarray(x), cfg)
    if r.shape[-1] == 0:
        return jnp.zeros(x.shape[:-1], dtype=x.dtype)
    return cfg.interaction_strength * jnp.sum(
        1.0 / jnp.sqrt(r**2 + cfg.softening**2), axis=-1
    )


def potential_energy(x: Array, cfg: SystemConfig) -> Array:
    return harmonic_potential(x, cfg) + interaction_energy(x, cfg)


def noninteracting_reference_energy(cfg: SystemConfig) -> float:
    """Exact g=0 energy for occupied 1D HO orbitals with fixed spin sectors."""
    up = sum((n + 0.5) * cfg.omega for n in range(cfg.n_up))
    down = sum((n + 0.5) * cfg.omega for n in range(cfg.n_down))
    return float(up + down)
