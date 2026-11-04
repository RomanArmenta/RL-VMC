from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SystemInfo:
    n_up: int
    n_down: int
    omega: float
    g: float
    a: float

    @property
    def n_electrons(self) -> int:
        return self.n_up + self.n_down

    @property
    def oscillator_length(self) -> float:
        return 1.0 / (self.omega ** 0.5)

    @property
    def lambda_coupling(self) -> float:
        return self.g / (self.omega ** 0.5)

    @property
    def softening_dimensionless(self) -> float:
        return self.a * (self.omega ** 0.5)

    def as_dict(self):
        d = asdict(self)
        d.update(
            n_electrons=self.n_electrons,
            oscillator_length=self.oscillator_length,
            lambda_coupling=self.lambda_coupling,
            softening_dimensionless=self.softening_dimensionless,
        )
        return d


@dataclass(frozen=True)
class GridConfig:
    x_min: float = -20.0
    x_max: float = 20.0
    n_grid: int = 401
    n_bins: int = 200
    r_max: float = 10.0
    pin_halfwidth: float = 0.12
    small_r_cut: float | None = None

    def validate(self):
        if not self.x_max > self.x_min:
            raise ValueError("x_max must exceed x_min")
        if self.n_grid < 21 or self.n_bins < 10:
            raise ValueError("grid/bin counts are too small")
        if self.r_max <= 0 or self.pin_halfwidth <= 0:
            raise ValueError("r_max and pin_halfwidth must be positive")
