from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class PhysicalSystem:
    """
    container for the static properties of our physical system. Ive call it "static_state".
    """
    name: str
    n_elec: int
    n_up: int
    n_dn: int
    coords: jnp.ndarray
    charges: jnp.ndarray
    charge: int = 0

    @property
    def spin(self):
        """Total spin"""
        return self.n_up - self.n_dn