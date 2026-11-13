from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_vmc.config import System1DConfig


class Harmonic1DHamiltonian:
    """Hamiltoniano para N electrones interactuantes en un pozo armónico 1D."""

    def __init__(self, cfg: System1DConfig):
        self.cfg = cfg
        self.n_elec = cfg.n_elec
        self.n_up = cfg.n_up
        self.n_dn = cfg.n_dn
        self.omega = cfg.omega
        self.g = cfg.interaction_strength
        self.softening = cfg.softening

    def noninteracting_reference_energy(self) -> float:
        """Energía exacta del estado base sin interacción (g=0) en el pozo armónico 1D."""
        e_up = sum(self.omega * (n + 0.5) for n in range(self.n_up))
        e_dn = sum(self.omega * (n + 0.5) for n in range(self.n_dn))
        return float(e_up + e_dn)

    def harmonic_potential(self, walkers: jax.Array) -> jax.Array:
        """Potencial de confinamiento armónico V_trap = 0.5 * omega^2 * sum_i x_i^2."""
        x = walkers[..., 0] if walkers.ndim == 3 else walkers
        return 0.5 * (self.omega ** 2) * jnp.sum(x ** 2, axis=-1)

    def interaction_energy(self, walkers: jax.Array) -> jax.Array:
        """Energía de interacción repulsiva soft-Coulomb entre todos los pares i < j."""
        x = walkers[..., 0] if walkers.ndim == 3 else walkers
        if self.n_elec < 2 or self.g == 0.0:
            return jnp.zeros(x.shape[0], dtype=x.dtype)

        idx_i, idx_j = jnp.triu_indices(self.n_elec, k=1)
        dx = x[:, idx_i] - x[:, idx_j]
        pair_v = self.g / jnp.sqrt(dx ** 2 + self.softening ** 2)
        return jnp.sum(pair_v, axis=-1)

    def potential_energy(self, walkers: jax.Array) -> jax.Array:
        """Energía potencial total V(X) = V_trap(X) + V_int(X)."""
        return self.harmonic_potential(walkers) + self.interaction_energy(walkers)

    def local_energy_and_components(
        self,
        ansatz_apply,
        params_psi,
        walkers: jax.Array,
        chunk_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Calcula la energía local E_L(X), energía cinética T_L(X) y potencial V_L(X).
        walkers: shape (batch, n_elec, 1)
        Si `chunk_size` es menor que `batch`, evalúa la cinética por bloques mediante `jax.lax.scan`
        para acotar la memoria pico durante el cálculo del Hessiano en sistemas con N grande.
        Retorna: (e_loc, t_loc, v_loc), cada uno de shape (batch,)
        """
        v_loc = self.potential_energy(walkers)

        def single_kinetic(w_single: jax.Array) -> jax.Array:
            x_flat = w_single.reshape(-1)

            def log_psi_flat(x_vec: jax.Array) -> jax.Array:
                w_reshaped = x_vec.reshape(1, self.n_elec, 1)
                val = ansatz_apply(params_psi, w_reshaped)
                return jnp.squeeze(val)

            grad_log = jax.grad(log_psi_flat)(x_flat)
            hess_log = jax.hessian(log_psi_flat)(x_flat)
            lapl_log = jnp.trace(hess_log)
            return -0.5 * (lapl_log + jnp.sum(grad_log ** 2))

        batch = walkers.shape[0]
        if chunk_size is not None and 0 < chunk_size < batch:
            n_full = batch // chunk_size
            rem = batch % chunk_size
            blocks = walkers[: n_full * chunk_size].reshape(
                (n_full, chunk_size) + walkers.shape[1:]
            )
            _, t_blocks = jax.lax.scan(
                lambda _, w_blk: (None, jax.vmap(single_kinetic)(w_blk)),
                None,
                blocks,
            )
            t_loc = t_blocks.reshape(-1)
            if rem > 0:
                t_rem = jax.vmap(single_kinetic)(walkers[n_full * chunk_size :])
                t_loc = jnp.concatenate([t_loc, t_rem], axis=0)
        else:
            t_loc = jax.vmap(single_kinetic)(walkers)

        e_loc = t_loc + v_loc
        return e_loc, t_loc, v_loc

    def local_energy(
        self,
        ansatz_apply,
        params_psi,
        walkers: jax.Array,
        chunk_size: int | None = None,
    ) -> jax.Array:
        e_loc, _, _ = self.local_energy_and_components(
            ansatz_apply, params_psi, walkers, chunk_size=chunk_size
        )
        return e_loc
