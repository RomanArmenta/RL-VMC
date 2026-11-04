"""
rl_vmc/core/optimizers.py
Constructores modulares de optimizadores Optax con soporte para Learning Rate Schedulers.
Permite encender y apagar el scheduler dinámicamente para todos los modelos entrenables
(Ansatz VMC, Red Actor de RL y Redes Críticas de RL).
"""
from __future__ import annotations

import optax
from typing import Any, Callable, Union

ScalarOrSchedule = Union[float, Callable[[Any], Any]]


def build_lr_schedule(
    base_lr: float,
    total_steps: int,
    use_scheduler: bool = True,
    scheduler_type: str = "cosine",
    warmup_steps: int = 100,
    min_ratio: float = 0.05,
) -> ScalarOrSchedule:
    """Construye un scheduler de learning rate de Optax o retorna un escalar constante.

    Args:
        base_lr: Learning rate pico o base.
        total_steps: Número total estimado de pasos de optimización.
        use_scheduler: Si es False, retorna base_lr como float constante.
        scheduler_type: 'cosine', 'exponential', 'linear' o 'constant'.
        warmup_steps: Pasos de calentamiento lineal inicial.
        min_ratio: Fracción mínima del learning rate final (default 0.05 = 5%).

    Returns:
        ScalarOrSchedule: Escalar constante o función de decaimiento compatible con Optax.
    """
    if not use_scheduler or scheduler_type == "constant" or total_steps <= 1:
        return float(base_lr)

    end_lr = float(base_lr * min_ratio)
    eff_warmup = min(max(1, warmup_steps), max(1, total_steps // 10))
    decay_steps = max(1, total_steps - eff_warmup)

    if scheduler_type == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=base_lr * 0.01,
            peak_value=base_lr,
            warmup_steps=eff_warmup,
            decay_steps=decay_steps,
            end_value=end_lr,
        )
    elif scheduler_type == "exponential":
        trans_steps = max(1, decay_steps // 4)
        return optax.exponential_decay(
            init_value=base_lr,
            transition_steps=trans_steps,
            decay_rate=0.7,
            end_value=end_lr,
        )
    elif scheduler_type == "linear":
        return optax.linear_schedule(
            init_value=base_lr,
            end_value=end_lr,
            transition_steps=total_steps,
        )
    else:
        return float(base_lr)
