from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from functools import partial
from src.system import PhysicalSystem

class _BaseHamiltonian(ABC):
    """
    abstract class for Hamiltonians in VMC
    implements the general computation of kinetic energy using JAX

    DEPRICATED FOR BETTER DIM CURSE MANAGEMENT  

    """

    def __init__(self, system: PhysicalSystem):
        self.system = system

    @abstractmethod
    def potential_energy(self, r_elec):
        """
        computes the potential energy V(R). Must be implemented in the inherit class
        """
        pass

    def kinetic_energy(self, ansatz_apply, params, r_elec):
        """
        computes the local kinetic energy using automatic differentiation 
        E_k = -0.5 * ( Laplacian( log_psi ) + |Grad ( log_psi ) |^2 )
        """

        # aux function for single walker
        def kinetic_single(r_single):
            def f(x):
                return ansatz_apply(params, x[None, ...])[0]
            
            grad_val = jax.grad(f)(r_single)
            grad_sq = jnp.sum(grad_val**2)

            # shape: (N_e, 3, N_e, 3)
            hess_val = jax.jacfwd(jax.jacrev(f))(r_single)

            # reshape to (N_e * 3, N_e * 3) and take its trace
            lap_val = jnp.trace(hess_val.reshape(self.system.n_elec * 3, self.system.n_elec * 3))

            return -0.5 * (lap_val + grad_sq)

        return jax.vmap(kinetic_single)(r_elec)
    
    def local_energy(self, ansatz_apply, params, r_elec):
        """
        returns the total local energy
        """
        k_en = self.kinetic_energy(ansatz_apply, params, r_elec)
        v_en = self.potential_energy(r_elec)
        return k_en + v_en
    
class BaseHamiltonian(ABC):
    """
    abstract class for Hamiltonians in VMC
    implements the general computation of kinetic energy using JAX
    """

    def __init__(self, system: PhysicalSystem):
        self.system = system

    @abstractmethod
    def potential_energy(self, r_elec):
        """
        computes the potential energy V(R). Must be implemented in the inherit class
        """
        pass

    def kinetic_energy(self, ansatz_apply, params, r_elec):
        """
        computes the local kinetic energy using automatic differentiation 
        E_k = -0.5 * ( Laplacian( log_psi ) + |Grad ( log_psi ) |^2 )
        """

        # aux function for single walker
        def kinetic_single(r_single):
            def f(x):
                return ansatz_apply(params, x[None, ...])[0]
            
            grad_f = jax.grad(f)
            grad_val = grad_f(r_single)
            grad_sq = jnp.sum(grad_val**2)

            n_e = self.system.n_elec
            dim = self.system.dim
            dim_total = n_e * dim
            
            basis = jnp.eye(dim_total).reshape(dim_total, n_e, dim)

            def scan_body(carry, v):
                _, hvp = jax.jvp(grad_f, (r_single,), (v,))
                
                diag_element = jnp.sum(hvp * v)
                return carry + diag_element, None
                
            lap_val, _ = jax.lax.scan(scan_body, 0.0, basis)

            return -0.5 * (lap_val + grad_sq)
            
        return jax.vmap(kinetic_single)(r_elec)
    
    def local_energy(self, ansatz_apply, params, r_elec):
        """
        returns the total local energy
        """
        k_en = self.kinetic_energy(ansatz_apply, params, r_elec)
        v_en = self.potential_energy(r_elec)
        return k_en + v_en
    
class FullCoulombHamiltonian(BaseHamiltonian):
    """
    exact molecular hamiltonian assuming static nuclei (Born - Oppenheimer)
    """
    def __init__(self, system):
        super().__init__(system)

        # pre-compute the nuclei-nuclei repulsion
        self.V_NN = self._compute_nuclear_repulsion()

    def _compute_nuclear_repulsion(self):
        """computes V_NN, and returns 0.0 if its only a single atom"""

        Z = self.system.charges
        R = self.system.coords

        # if there is more than 1 atom, compute its potential energy
        if len(Z) > 1:
            diff_NN = R[:, None, :] - R[None, :, :]
            dist_NN = jnp.linalg.norm(diff_NN, axis=-1)
            # upper triangular matrix so we dont double count nor auto-interactions
            mask_NN = jnp.triu(jnp.ones((len(Z), len(Z))), k=1) # k=1 for not consider the diagonal
            dist_safe_NN = dist_NN + jnp.eye(len(Z))
            V_NN = jnp.sum((Z[:, None] * Z[None, :] / dist_safe_NN) * mask_NN)
            return V_NN
        return 0.0
    
    def potential_energy(self, r_elec):
        """
        computes V_en (atraction) and V_ee (repulsion) for a walkers batch
        r_elec shape: [batch_size, N_elec, 3]
        """

        # V_en
        diff_en = r_elec[:, :, None, :] - self.system.coords[None, None, :, :]
        dist_en = jnp.linalg.norm(diff_en, axis=-1) + 1e-12
        V_en = -jnp.sum(self.system.charges[None, None, :] / dist_en, axis = (1, 2))

        # V_ee
        diff_ee = r_elec[:, :, None, :] - r_elec[:, None, :, :]
        dist_ee = jnp.linalg.norm(diff_ee, axis=-1) + 1e-12
        mask_ee = jnp.triu(jnp.ones((self.system.n_elec, self.system.n_elec)), k=1)
        dist_ee_safe = dist_ee + jnp.eye(self.system.n_elec) * 1e6 # avoid division by zero in diag

        V_ee = jnp.sum((1.0 / dist_ee_safe) * mask_ee[None, :, :], axis=(1, 2))

        return V_en + V_ee + self.V_NN

class Harmonic1DHamiltonian(BaseHamiltonian):
    def potential_energy(self, r_elec):
        """
        r_elec shape: [batch, N_elec, 1]
        """

        # get spatial dimension -> shape: [batch, N_elec]
        x = r_elec[..., 0]

        # harmonic potential
        v_harm = 0.5 * (self.system.omega**2) * jnp.sum(x**2, axis = -1)

        # interaction
        diff_ee = x[:, :, None] - x[:, None, :]
        dist_sq = diff_ee**2 + self.system.softening**2
        mask_ee = jnp.triu(jnp.ones((self.system.n_elec, self.system.n_elec)), k=1)

        v_int = self.system.interaction_strength * jnp.sum(
            (1.0 / jnp.sqrt(dist_sq)) * mask_ee, axis = (1, 2)
        )
        return v_harm + v_int