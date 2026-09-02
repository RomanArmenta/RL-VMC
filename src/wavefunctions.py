import os
import os
import deepqmc
from abc import ABC, abstractmethod
from src.system import PhysicalSystem
from deepqmc.molecule import Molecule
from deepqmc.app import instantiate_ansatz
from deepqmc.hamil import MolecularHamiltonian
from deepqmc.types import PhysicalConfiguration
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from src.physics import SystemConfig
from src.wavefunction import log_abs_psi, init_wavefunction_params
from src.orbitals import OrbitalConfig
import jax
import jax.numpy as jnp

# do i have to import the PhysicalSystem class?

class BaseAnsatz(ABC):
    """
    universal interface for ansatz in our VMC implementation
    """

    def __init__(self, system: PhysicalSystem):
        self.system = system

    @abstractmethod
    def init(self, rng_key, batch_walkers):
        """
        Initializes and retunrs the parameters of the wavefunction
        """
        pass

    @abstractmethod
    def apply(self, params, walkers):
        """
        Evaluates the natural logarithm of the wavefunctions norm: log |Psi|
        must return a tensor of shape (batch_size,) 
        """
        pass

class DeepQMCAnsatz_(BaseAnsatz):
    """
    wrapper for DeepQMC ansatz.
    """
    def __init__(self, system: PhysicalSystem, config_name: str = "default"):
        super().__init__(system)

        self.mol = Molecule(
            coords = self.system.coords,
            charges = self.system.charges,
            charge = self.system.charge,
            spin = self.system.spin
        )

        self.hamil = MolecularHamiltonian(mol = self.mol)

        deepqmc_dir = os.path.dirname(deepqmc.__file__)
        config_dir = os.path.join(deepqmc_dir, "conf", "ansatz")

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_name)
            _ansatz_factory = instantiate(cfg, _recursive_=True, _convert_="all")

        self.ansatz = instantiate_ansatz(self.hamil, _ansatz_factory)

        self._batched_apply = jax.vmap(self.ansatz.apply, in_axes = (None, 0))

    def init(self, rng_key, batch_walkers):
        """
        generates weights from the model evaluating a single test walker
        """

        single_walker = batch_walkers[0]

        phys_conf = PhysicalConfiguration(
            R = self.system.coords,
            r = single_walker,
            mol_idx = jnp.array(0, dtype=jnp.int32)
        )

        return self.ansatz.init(rng_key, phys_conf)
    
    def apply(self, params, walkers):
        """
        evaluates a batch of walkers and returns a 1D tensor of log |Psi|
        """
        batch_size = walkers.shape[0]

        # clone into [batch_size, N_nuclei, 3]
        R_batch = jnp.tile(self.system.coords, (batch_size, 1, 1))

        # all walkers in the same molecule (index 0)
        mol_idx_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        phys_conf_batch = PhysicalConfiguration(
            R = R_batch,
            r = walkers,
            mol_idx = mol_idx_batch
        )

        out = self._batched_apply(params, phys_conf_batch)
        return out.log

class DeepQMCAnsatz(BaseAnsatz):
    """
    wrapper for DeepQMC ansatz.
    """
    def __init__(self, system: PhysicalSystem, config_name: str = "default", n_determinants: int = 10, backflow_transform: str = 'mult'):
        super().__init__(system)

        self.mol = Molecule(
            coords = self.system.coords,
            charges = self.system.charges,
            charge = self.system.charge,
            spin = self.system.spin
        )

        self.hamil = MolecularHamiltonian(mol = self.mol)

        deepqmc_dir = os.path.dirname(deepqmc.__file__)
        config_dir = os.path.join(deepqmc_dir, "conf", "ansatz")

        # Manejamos el string "none" para que Hydra lo interprete como objeto nulo (apaga el backflow)
        bf_val = "null" if backflow_transform.lower() == "none" else f"'{backflow_transform}'"
        
        # Inyectamos los argumentos directamente al compose de Hydra
        overrides = [
            f"n_determinants={n_determinants}",
            f"backflow_transform={bf_val}"
        ]

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_name, overrides=overrides)
            _ansatz_factory = instantiate(cfg, _recursive_=True, _convert_="all")

        self.ansatz = instantiate_ansatz(self.hamil, _ansatz_factory)

        self._batched_apply = jax.vmap(self.ansatz.apply, in_axes = (None, 0))

    def init(self, rng_key, batch_walkers):
        """
        generates weights from the model evaluating a single test walker
        """
        single_walker = batch_walkers[0]

        phys_conf = PhysicalConfiguration(
            R = self.system.coords,
            r = single_walker,
            mol_idx = jnp.array(0, dtype=jnp.int32)
        )

        return self.ansatz.init(rng_key, phys_conf)
    
    def apply(self, params, walkers):
        """
        evaluates a batch of walkers and returns a 1D tensor of log |Psi|
        """
        batch_size = walkers.shape[0]

        # clone into [batch_size, N_nuclei, 3]
        R_batch = jnp.tile(self.system.coords, (batch_size, 1, 1))

        # all walkers in the same molecule (index 0)
        mol_idx_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        phys_conf_batch = PhysicalConfiguration(
            R = R_batch,
            r = walkers,
            mol_idx = mol_idx_batch
        )

        out = self._batched_apply(params, phys_conf_batch)
        return out.log


class Ansatz1D_Huziel(BaseAnsatz):
    def __init__(self, system, wf_cfg, orbital_cfg, correlation_cfg):
        super().__init__(system)
        self.wf_cfg = wf_cfg
        self.orbital_cfg = orbital_cfg
        self.correlation_cfg = correlation_cfg

        # adapt modular system to new dataclass
        self.sys_cfg = SystemConfig(
            n_electrons=system.n_elec,
            n_up = system.n_up,
            omega = system.omega,
            interaction_strength=system.interaction_strength,
            softening=system.softening
        )

    def init(self, rng_key, batch_walkers):
        # we need to init the parameters
        return init_wavefunction_params(
            rng_key, self.sys_cfg, self.wf_cfg, self.orbital_cfg, self.correlation_cfg
        )

    def apply(self, params, walkers):
        # walkers shape: [batch, N_elec, 1]
        # we want [N] per molecule
        # get rid of extra spatial dimension -> [batch, N_elec]
        
        x_1d = walkers[..., 0]

        def single_apply(x):
            return log_abs_psi(params, x, self.sys_cfg, self.wf_cfg, self.orbital_cfg, self.correlation_cfg)

        return jax.vmap(single_apply)(x_1d)

class Ansatz1D_Backflow(BaseAnsatz):
    """
    Ansatz modular que incorpora Backflow sobre orbitales de Oscilador Armónico.
    Las coordenadas se deforman x_eff = x + xi(x) antes de entrar a los determinantes.
    """
    def __init__(self, system, wf_cfg, orbital_cfg, correlation_cfg):
        super().__init__(system)
        self.wf_cfg = wf_cfg
        # Forzamos el uso de Backflow sin importar el string del parser
        self.orbital_cfg = OrbitalConfig(**{**orbital_cfg.__dict__, "model": "backflow_ho"})
        self.correlation_cfg = correlation_cfg

        self.sys_cfg = SystemConfig(
            n_electrons=system.n_elec,
            n_up=system.n_up,
            omega=system.omega,
            interaction_strength=system.interaction_strength,
            softening=system.softening
        )

    def init(self, rng_key, batch_walkers):
        return init_wavefunction_params(
            rng_key, self.sys_cfg, self.wf_cfg, self.orbital_cfg, self.correlation_cfg
        )

    def apply(self, params, walkers):
        # Aplanar la dimensión espacial extra [batch, N_elec, 1] -> [batch, N_elec]
        x_1d = walkers[..., 0]

        def single_apply(x):
            return log_abs_psi(params, x, self.sys_cfg, self.wf_cfg, self.orbital_cfg, self.correlation_cfg)

        return jax.vmap(single_apply)(x_1d)