from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class PhysicalSystem:
    """
    container for the static properties of our physical system. Ive call it "static_state" before.
    """
    name: str
    n_elec: int
    n_up: int
    n_dn: int
    coords: jnp.ndarray
    charges: jnp.ndarray
    charge: int = 0
    dim: int = 3

    @property
    def spin(self):
        """Total spin"""
        return self.n_up - self.n_dn

@dataclass
class Harmonic_1D(PhysicalSystem):
    """
    Harmonic potential in one dimension
    """
    omega: float = 1.0
    interaction_strength: float = 1.0
    softening: float = 0.25
    dim: int = 1