from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UHFConfig:
    n_up: int = 2
    n_down: int = 2
    omega: float = 1.0
    g: float = 1.0
    softening: float = 0.25
    n_basis: int = 12
    quadrature_order: int = 80
    max_iter: int = 200
    energy_tol: float = 1e-10
    density_tol: float = 1e-8
    mixing: float = 0.35
    diis: bool = True
    diis_start: int = 3
    diis_size: int = 8
    level_shift: float = 0.0

    @property
    def n_electrons(self):
        return self.n_up + self.n_down

    def validate(self):
        if min(self.n_up, self.n_down) < 0 or self.n_electrons < 1:
            raise ValueError("spin populations must be nonnegative and N >= 1")
        if self.n_basis < max(self.n_up, self.n_down):
            raise ValueError("n_basis must accommodate both spin populations")
        if self.omega <= 0 or self.softening < 0:
            raise ValueError("omega and softening must be positive")
        if self.g < 0 or not 0 < self.mixing <= 1:
            raise ValueError("require g >= 0 and 0 < mixing <= 1")
        if self.quadrature_order < self.n_basis + 2:
            raise ValueError("quadrature_order is too small for the basis")
        return self

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CASSCFConfig(UHFConfig):
    """State-specific complete-active-space SCF configuration."""
    n_core: int = 0
    n_active_orbitals: int = 4
    casscf_max_iter: int = 500
    casscf_energy_tol: float = 1e-4
    orbital_gradient_tol: float = 1e-5
    max_ci_dimension: int = 20000
    n_final_determinants: int = 0

    @property
    def n_active_up(self):
        return self.n_up - self.n_core

    @property
    def n_active_down(self):
        return self.n_down - self.n_core

    def validate(self):
        super().validate()
        if not 0 <= self.n_core <= min(self.n_up, self.n_down):
            raise ValueError("n_core must be between 0 and min(n_up, n_down)")
        if self.n_active_orbitals < max(self.n_active_up, self.n_active_down):
            raise ValueError("active space cannot hold the active electrons")
        if self.n_core + self.n_active_orbitals > self.n_basis:
            raise ValueError("core plus active orbitals exceeds n_basis")
        if self.casscf_max_iter < 1 or self.max_ci_dimension < 1:
            raise ValueError("CASSCF iteration and CI dimension limits must be positive")
        if self.n_final_determinants < 0:
            raise ValueError("n_final_determinants must be 0 (all) or positive")
        return self
