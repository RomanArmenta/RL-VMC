from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from typing import Literal, Optional


@dataclass
class System1DConfig:
    """Parámetros del sistema físico 1D."""
    n_elec: int = 4
    n_up: int = 2
    n_dn: int = 2
    omega: float = 0.1
    interaction_strength: float = 1.0
    softening: float = 0.1


@dataclass
class Ansatz1DConfig:
    """Parámetros de la función de onda (Ansatz) 1D."""
    ansatz_type: Literal["slater_jastrow_1d", "casscf_npz", "multi_sjb", "multi_slater_jastrow"] = "slater_jastrow_1d"
    wf_model: str = "slater_jastrow"
    orbital_model: Literal["ho_fixed", "ho_variational", "gaussian_basis", "backflow_ho"] = "backflow_ho"
    correlation_model: Literal["none", "jastrow", "spin_rbf", "spin_rbf_jastrow", "spin_rbf_threebody"] = "spin_rbf_jastrow"
    backflow_components: int = 8
    backflow_r_max: float = 5.0
    backflow_width: float = 0.7
    backflow_envelope_length: float = 3.0
    casscf_npz_path: Optional[str] = None
    casscf_trainable: bool = False


@dataclass
class Proposal1DConfig:
    """Configuración del kernel de propuesta (Clásico o Actor Neuronal)."""
    proposal_type: Literal["metropolis", "mala", "hmc", "neural_deepset", "neural_flat"] = "mala"
    proposal_mode: Literal["residual_mala", "scale_only", "full_gaussian"] = "residual_mala"
    tau: float = 0.01
    rw_sigma: float = 0.5
    hmc_leapfrog_steps: int = 5
    hmc_step_size: float = 0.05
    actor_amplitude: float = 0.15
    sigma_action_scale: float = 0.2
    max_drift_clip: float = 10.0
    action_rev_limit: float = 1.0
    scale_clip_min: float = -5.0
    scale_clip_max: float = 1.0
    force_scale: float = 5.0
    n_scale: float = 10.0
    r_scale: float = 5.0
    hidden_dim_local: int = 32
    hidden_dim_global: int = 16
    hidden_dim_messages: int = 32
    hidden_dim_actor: int = 64
    hidden_dim_critic: int = 64
    use_spatial_gating: bool = False
    spatial_gating_radius: float = 10.0
    legacy_drift_factor: bool = True
    parity_equivariant: bool = False
    unclip_sigma_action: bool = False


@dataclass
class Sampler1DConfig:
    """Configuración del motor Metropolis-Hastings y estabilidad numérica."""
    batch_size: int = 2048
    update_mode: Literal["all_electrons", "single_electron"] = "all_electrons"
    sweeps_per_step: int = 1
    n_burn_in: int = 2000
    init_mode: Literal[
        "symmetric_gaussian",
        "alternating_spin",
        "spin_ordered_offset",
        "randomized_wigner",
        "uniform",
    ] = "symmetric_gaussian"
    init_std: float = 3.0
    init_offset_base: float = 5.0
    init_offset_step: float = 1.0
    init_uniform_halfwidth: float = 0.0
    min_clip_grads_ansatz: float = -500.0
    max_clip_grads_ansatz: float = 500.0
    min_clip_log_ratio: float = -50.0
    max_clip_log_ratio: float = 50.0
    nan_clip_log_ratio: float = -50.0
    posinf_clip_log_ratio: float = 50.0
    neginf_clip_log_ratio: float = -50.0


@dataclass
class RL1DConfig:
    """Hiperparámetros del algoritmo de Reinforcement Learning y función de recompensa."""
    algorithm: Literal["td3", "reinforce"] = "td3"
    reward_type: Literal["cdlb", "log_esjd", "barker_jump", "kappa_jump"] = "cdlb"
    lr_actor: float = 1e-3
    lr_critic: float = 3e-4
    rl_gamma_td3: float = 0.99
    tau_polyak: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.15
    noise_clip: float = 0.35
    exp_noise_init: float = 0.20
    exp_noise_final: float = 0.00
    rl_replay_capacity: int = 100000
    rl_batch_size: int = 256
    penalty_factor: float = 0.1
    beta: float = 10.0
    gamma: float = 0.1
    kappa: float = 0.5
    reward_epsilon: float = 1e-3
    max_phys_jump_factor: float = 5.0
    staggered_critic_reset: bool = False
    use_reward_centering: bool = True
    reward_centering_lr: float = 0.01
    cdlb_exact_entropy: bool = False
    use_lr_scheduler: bool = True
    lr_scheduler_type: Literal["cosine", "exponential", "linear", "constant"] = "cosine"
    lr_warmup_steps: int = 100
    lr_min_ratio: float = 0.05


@dataclass
class Train1DConfig:
    """Configuración del régimen de entrenamiento (VMC, RL, Acoplado por bloques, Acoplado Online P4 o Solo Evaluación)."""
    mode: Literal["vmc_only", "rl_only", "coupled", "online_coupled", "eval_only"] = "vmc_only"
    seed: int = 42
    lr_ansatz: float = 5e-4
    use_lr_scheduler: bool = True
    lr_scheduler_type: Literal["cosine", "exponential", "linear", "constant"] = "cosine"
    lr_warmup_steps: int = 100
    lr_min_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    train_steps: int = 5000
    coupled_training_cycles: int = 8
    vmc_iterations_per_cycle: int = 1250
    rl_sweeps_per_cycle: int = 100
    online_nodal_warmup_steps: int = 250
    online_critic_warmup_steps: int = 250
    online_rl_freeze_tail_steps: int = 500
    online_rl_update_every: int = 1
    online_rl_updates_per_step: int = 1
    online_walkers_to_store: int = 64
    online_exp_noise: float = 0.0
    online_reset_critic_period: int = 0
    steps_for_best: int = 500
    energy_clip_min: float = -500.0
    energy_clip_max: float = 500.0
    sigma_energy_clip: float = 5.0
    n_print: int = 100
    n_log: int = 50


@dataclass
class Eval1DConfig:
    """Configuración de la fase de evaluación / producción MCMC con pesos congelados."""
    run_evaluation: bool = True
    n_eval_chains: int = 1
    eval_burn_in: int = 2000
    eval_init_mode: Literal[
        "same_as_sampler",
        "symmetric_gaussian",
        "alternating_spin",
        "spin_ordered_offset",
        "randomized_wigner",
        "uniform",
    ] = "same_as_sampler"
    measure_steps: int = 5000
    thin: int = 1
    jit_chunk_size: int = 1000
    eloc_chunk_size: int = 20000
    use_actor_in_eval: bool = True


@dataclass
class IOConfig:
    """Banderas de guardado y rutas de entrada/salida para el archivo unificado HDF5."""
    output_h5: str = "results/run_bundle.h5"
    results_out_dir: Optional[str] = None
    run_name: Optional[str] = None
    load_bundle_h5: Optional[str] = None
    load_legacy_dir: Optional[str] = None
    save_config: bool = True
    save_ansatz_weights: bool = True
    save_actor_weights: bool = True
    save_critic_weights: bool = True
    save_training_history: bool = True
    save_training_csv: bool = True
    save_production_walkers: bool = True
    save_walkers_max_batch: int = 64
    save_production_energies: bool = True
    save_production_energy_mode: Literal["step_mean", "full"] = "step_mean"
    save_production_walkers_dtype: Literal["float32", "float64"] = "float32"
    save_production_stats_dtype: Literal["float32", "float64"] = "float32"
    save_production_sampler_stats: bool = True
    export_legacy_files: bool = False
    save_summary_json: bool = True
    master_csv_path: Optional[str] = None
    csv_collect_dir: Optional[str] = "results_csv"
    reference_casscf_npz: Optional[str] = None
    uhf_search_root: str = "."
    pipeline_stage: str = "general"


@dataclass
class Experiment1DConfig:
    """Configuración maestra unificada para el pipeline 1D."""
    system: System1DConfig = field(default_factory=System1DConfig)
    ansatz: Ansatz1DConfig = field(default_factory=Ansatz1DConfig)
    proposal: Proposal1DConfig = field(default_factory=Proposal1DConfig)
    sampler: Sampler1DConfig = field(default_factory=Sampler1DConfig)
    rl: RL1DConfig = field(default_factory=RL1DConfig)
    train: Train1DConfig = field(default_factory=Train1DConfig)
    eval: Eval1DConfig = field(default_factory=Eval1DConfig)
    io: IOConfig = field(default_factory=IOConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment1DConfig":
        def _filter_fields(dc_cls, sub_dict):
            if not isinstance(sub_dict, dict):
                return dc_cls()
            valid = {f.name for f in dc_cls.__dataclass_fields__.values()}
            return dc_cls(**{k: v for k, v in sub_dict.items() if k in valid})

        return cls(
            system=_filter_fields(System1DConfig, data.get("system", {})),
            ansatz=_filter_fields(Ansatz1DConfig, data.get("ansatz", {})),
            proposal=_filter_fields(Proposal1DConfig, data.get("proposal", {})),
            sampler=_filter_fields(Sampler1DConfig, data.get("sampler", {})),
            rl=_filter_fields(RL1DConfig, data.get("rl", {})),
            train=_filter_fields(Train1DConfig, data.get("train", {})),
            eval=_filter_fields(Eval1DConfig, data.get("eval", {})),
            io=_filter_fields(IOConfig, data.get("io", {})),
        )


@dataclass
class PostProcessConfig:
    """Configuración y banderas para el módulo de post-procesamiento."""
    inputs: list[str] = field(default_factory=list)
    labels: Optional[list[str]] = None
    out_dir: str = "results/postprocess"
    reference_density_csv: Optional[str] = None
    casscf_npz_path: Optional[str] = None
    uhf_search_root: Optional[str] = "."
    auto_casscf_baseline: bool = True
    master_comparison_csv: Optional[str] = None
    csv_collect_dir: Optional[str] = "results_csv"
    system_tag: Optional[str] = None
    recompute_local_energy: bool = False
    sokal_window_c: float = 5.0
    max_acf_lag: int = 2000
    target_se_epsilon: float = 1e-3
    n_density_bins: int = 120
    density_x_max: float = 15.0
    compute_observables: bool = True
    compute_diagnostics: bool = True
    save_summary_csv: bool = True
    save_observables_csv: bool = True
    plot_training_metrics: bool = True
    plot_production_summary: bool = True
    plot_observables_comparison: bool = True
    plot_efficiency_bars: bool = True
    plot_time_to_solution: bool = True
    plot_walker_trace: bool = True
    plot_kl_divergence: bool = False
    plot_hyperparameter_evolution: bool = True
    top_k_models: int = 2
    walker_trace_top_k: int = 1
    n_eval_subblocks: int = 5
    figure_format: Literal["pdf", "png", "both"] = "png"
    dpi: int = 300
    title_prefix: str = ""
    custom_suptitle: str = ""
    no_titles: bool = False
    font_scale: float = 1.0
    line_width_scale: float = 1.0
    transparent_bg: bool = False


def build_experiment_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RL-VMC 1D: Pipeline Modular de Entrenamiento, Muestreo y Empaquetado HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Atajos de compatibilidad con scripts previos
    g_compat = parser.add_argument_group("0. Atajos Rápidos / Compatibilidad")
    g_compat.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["MH", "MALA", "HMC", "TD3", "REINFORCE"],
        help="Atajo opcional para preconfigurar proposal_type y train.mode (ej. 'MALA' o 'TD3').",
    )

    # 1. Sistema Físico
    g_sys = parser.add_argument_group("1. Parámetros del Sistema Físico 1D")
    g_sys.add_argument("--n_elec", type=int, default=4, help="Número total de electrones en el sistema.")
    g_sys.add_argument("--n_up", type=int, default=2, help="Número de electrones con espín arriba (up).")
    g_sys.add_argument("--n_dn", type=int, default=2, help="Número de electrones con espín abajo (down).")
    g_sys.add_argument("--omega", type=float, default=0.1, help="Frecuencia angular del pozo armónico confinante.")
    g_sys.add_argument("--interaction_strength", "--g", dest="interaction_strength", type=float, default=1.0, help="Fuerza de interacción repulsiva electrón-electrón (g).")
    g_sys.add_argument("--softening", type=float, default=0.1, help="Parámetro de suavizado (soft-Coulomb) a.")

    # 2. Ansatz
    g_ans = parser.add_argument_group("2. Arquitectura del Ansatz (Función de Onda)")
    g_ans.add_argument("--ansatz_type", type=str, default="slater_jastrow_1d", choices=["slater_jastrow_1d", "casscf_npz", "multi_sjb", "multi_slater_jastrow"], help="Familia del ansatz a utilizar.")
    g_ans.add_argument("--wf_model", type=str, default="slater_jastrow", help="Nombre de jerarquía del ansatz 1D.")
    g_ans.add_argument("--orbital_model", type=str, default="backflow_ho", choices=["ho_fixed", "ho_variational", "gaussian_basis", "backflow_ho"], help="Modelo de orbitales espaciales.")
    g_ans.add_argument("--correlation_model", type=str, default="spin_rbf_jastrow", choices=["none", "jastrow", "spin_rbf", "spin_rbf_jastrow", "spin_rbf_threebody"], help="Modelo del factor de correlación Jastrow/RBF.")
    g_ans.add_argument("--backflow_components", type=int, default=8, help="Número de componentes radiales en la transformación Backflow.")
    g_ans.add_argument("--backflow_r_max", type=float, default=5.0, help="Radio máximo de corte para el Backflow.")
    g_ans.add_argument("--backflow_width", type=float, default=0.7, help="Anchura de las funciones base radiales del Backflow.")
    g_ans.add_argument("--backflow_envelope_length", type=float, default=3.0, help="Longitud de envolvente exponencial del Backflow.")
    g_ans.add_argument("--casscf_npz_path", type=str, default=None, help="Ruta a archivo .npz si ansatz_type='casscf_npz'.")
    g_ans.add_argument("--casscf_trainable", action=argparse.BooleanOptionalAction, default=False, help="Habilita la optimización variacional de orbitales y CI en CASSCF.")

    # 3. Propuesta / Actor
    g_prop = parser.add_argument_group("3. Política de Propuesta (Clásica o Actor Neuronal)")
    g_prop.add_argument("--proposal_type", type=str, default="mala", choices=["metropolis", "mala", "hmc", "neural_deepset", "neural_flat"], help="Tipo de actor/propuesta para generar transiciones.")
    g_prop.add_argument("--proposal_mode", type=str, default="residual_mala", choices=["residual_mala", "scale_only", "full_gaussian"], help="Modo en que la salida del Actor modifica la propuesta.")
    g_prop.add_argument("--tau", type=float, default=0.01, help="Tamaño de paso temporal discreto para MALA / RL-MALA.")
    g_prop.add_argument("--rw_sigma", type=float, default=0.5, help="Desviación estándar del salto para Metropolis Random Walk.")
    g_prop.add_argument("--hmc_leapfrog_steps", type=int, default=5, help="Número de pasos leapfrog en Hamiltonian Sampling (HMC).")
    g_prop.add_argument("--hmc_step_size", type=float, default=0.05, help="Tamaño de paso del integrador simpléctico en HMC.")
    g_prop.add_argument("--actor_amplitude", type=float, default=0.15, help="Amplitud máxima de la corrección de deriva del Actor neuronal.")
    g_prop.add_argument("--sigma_action_scale", type=float, default=0.2, help="Escala máxima de la salida tanh para modificar la varianza.")
    g_prop.add_argument("--max_drift_clip", type=float, default=10.0, help="Recorte absoluto de la deriva de Langevin.")
    g_prop.add_argument("--action_rev_limit", type=float, default=1.0, help="Límite de recorte de la acción al evaluar la transición inversa.")
    g_prop.add_argument("--scale_clip_min", type=float, default=-5.0, help="Cota inferior para el exponente de escala de varianza.")
    g_prop.add_argument("--scale_clip_max", type=float, default=1.0, help="Cota superior para el exponente de escala de varianza.")
    g_prop.add_argument("--force_scale", type=float, default=5.0, help="Escala de normalización de la fuerza cuántica en el estado DeepSet.")
    g_prop.add_argument("--n_scale", type=float, default=10.0, help="Escala de normalización del número de partículas en el estado DeepSet.")
    g_prop.add_argument("--r_scale", type=float, default=5.0, help="Escala de normalización de distancias inter-electrónicas en el estado DeepSet.")
    g_prop.add_argument("--use_spatial_gating", action=argparse.BooleanOptionalAction, default=False, help="Activa la máscara espacial suave para atenuar la deriva del actor lejos del centro.")
    g_prop.add_argument("--spatial_gating_radius", type=float, default=10.0, help="Radio característico (Bohr) para el spatial gating.")
    g_prop.add_argument("--legacy_drift_factor", action=argparse.BooleanOptionalAction, default=True, help="Si es True usa drift=tau*grad_log_psi (compatible con pesos previos); si es False usa drift=2*tau*grad_log_psi.")
    g_prop.add_argument("--parity_equivariant", action=argparse.BooleanOptionalAction, default=False, help="Si es True impone equivarianza exacta de paridad en FlaxTD3Actor: mu(-x,-F)=-mu(x,F) y sigma(-x,-F)=+sigma(x,F). Si es False usa el actor original.")
    g_prop.add_argument("--unclip_sigma_action", action=argparse.BooleanOptionalAction, default=False, help="Si es True desbloquea el techo duro [-1,1] sobre el canal a_sigma (usando [scale_clip_min, scale_clip_max]) en forward_action y TD3 target smoothing. Si es False usa el recorte legado [-1,1].")

    # 4. Motor MCMC y Estabilidad
    g_samp = parser.add_argument_group("4. Motor Metropolis-Hastings y Estabilidad Numérica")
    g_samp.add_argument("--batch_size", type=int, default=2048, help="Número de caminantes (walkers) paralelos.")
    g_samp.add_argument("--n_walkers", dest="batch_size", type=int, help="Alias de --batch_size.")
    g_samp.add_argument("--update_mode", type=str, default="all_electrons", choices=["all_electrons", "single_electron"], help="Actualización simultánea de todos los electrones o secuencial uno por uno.")
    g_samp.add_argument("--sweeps_per_step", type=int, default=1, help="Barridos MCMC ejecutados entre cada actualización de gradiente VMC.")
    g_samp.add_argument("--n_ter", "--n_burn_in", dest="n_burn_in", type=int, default=2000, help="Pasos de termalización previos al entrenamiento.")
    g_samp.add_argument("--init_mode", type=str, default="symmetric_gaussian", choices=["symmetric_gaussian", "alternating_spin", "spin_ordered_offset", "randomized_wigner", "uniform"], help="Modo de inicialización espacial de los caminantes.")
    g_samp.add_argument("--init_std", type=float, default=3.0, help="Desviación estándar inicial (Bohr) para init_mode='symmetric_gaussian'.")
    g_samp.add_argument("--init_offset_base", type=float, default=5.0, help="Posición base inicial (Bohr) para init_mode='spin_ordered_offset'.")
    g_samp.add_argument("--init_offset_step", type=float, default=1.0, help="Separación entre electrones (Bohr) para init_mode='spin_ordered_offset'.")
    g_samp.add_argument("--init_uniform_halfwidth", type=float, default=0.0, help="Semiancho de la caja [-L, L] (Bohr) para init_mode='uniform'. Si es 0.0, se calcula automáticamente según la dispersión cuántica.")
    g_samp.add_argument("--min_clip_grads_ansatz", type=float, default=-500.0, help="Cota inferior para recortar el gradiente espacial de log|Psi|.")
    g_samp.add_argument("--max_clip_grads_ansatz", type=float, default=500.0, help="Cota superior para recortar el gradiente espacial de log|Psi|.")
    g_samp.add_argument("--min_clip_log_ratio", type=float, default=-50.0, help="Cota inferior del log-ratio de aceptación de Metropolis.")
    g_samp.add_argument("--max_clip_log_ratio", type=float, default=50.0, help="Cota superior del log-ratio de aceptación de Metropolis.")
    g_samp.add_argument("--nan_clip_log_ratio", type=float, default=-50.0, help="Valor de reemplazo en caso de NaN en el log-ratio.")
    g_samp.add_argument("--posinf_clip_log_ratio", type=float, default=50.0, help="Valor de reemplazo en caso de +Inf en el log-ratio.")
    g_samp.add_argument("--neginf_clip_log_ratio", type=float, default=-50.0, help="Valor de reemplazo en caso de -Inf en el log-ratio.")

    # 5. Reinforcement Learning
    g_rl = parser.add_argument_group("5. Algoritmo de Reinforcement Learning y Recompensa")
    g_rl.add_argument("--rl_algorithm", type=str, default="td3", choices=["td3", "reinforce"], help="Algoritmo RL para entrenar el Actor.")
    g_rl.add_argument("--reward_type", type=str, default="cdlb", choices=["cdlb", "log_esjd", "barker_jump", "kappa_jump", "displacement"], help="Fórmula de recompensa para evaluar propuestas MCMC.")
    g_rl.add_argument("--reward_mode", dest="reward_type", type=str, help="Alias de --reward_type.")
    g_rl.add_argument("--lr_actor", type=float, default=1e-3, help="Tasa de aprendizaje (Adam) para la red Actor.")
    g_rl.add_argument("--lr_critic", type=float, default=3e-4, help="Tasa de aprendizaje (Adam) para las redes Críticas (TD3).")
    g_rl.add_argument("--rl_gamma_td3", type=float, default=0.99, help="Factor de descuento temporal gamma en la ecuación de Bellman (TD3).")
    g_rl.add_argument("--tau_polyak", type=float, default=0.005, help="Coeficiente de actualización suave (Polyak) para redes objetivo TD3.")
    g_rl.add_argument("--policy_delay", type=int, default=2, help="Frecuencia de actualización del Actor respecto a los Críticos en TD3.")
    g_rl.add_argument("--target_noise", type=float, default=0.15, help="Desviación estándar del ruido de suavizado de política objetivo TD3.")
    g_rl.add_argument("--noise_clip", type=float, default=0.35, help="Cota de recorte del ruido de suavizado en TD3.")
    g_rl.add_argument("--exp_noise_init", type=float, default=0.20, help="Ruido de exploración inicial en las acciones durante entrenamiento RL.")
    g_rl.add_argument("--exp_noise_final", type=float, default=0.00, help="Ruido de exploración final en las acciones durante entrenamiento RL.")
    g_rl.add_argument("--rl_replay_capacity", type=int, default=100000, help="Capacidad máxima del Replay Buffer en TD3.")
    g_rl.add_argument("--rl_batch_size", type=int, default=256, help="Tamaño de minibatch muestreado del Replay Buffer en TD3.")
    g_rl.add_argument("--penalty_factor", type=float, default=0.1, help="Factor de escala lambda para la recompensa CDLB o penalización de rechazo.")
    g_rl.add_argument("--beta", type=float, default=10.0, help="Peso del término de distancia en recompensas ESJD/Barker.")
    g_rl.add_argument("--gamma", type=float, default=0.1, help="Peso secundario de aceptación/fuerza en la función de recompensa.")
    g_rl.add_argument("--kappa", type=float, default=0.5, help="Coeficiente del término alpha*(1-alpha) en recompensa kappa_jump.")
    g_rl.add_argument("--max_phys_jump_factor", type=float, default=5.0, help="Múltiplo de sqrt(tau) para acotar la distancia máxima premiada.")
    g_rl.add_argument("--staggered_critic_reset", action=argparse.BooleanOptionalAction, default=False, help="Reinicia alternadamente uno de los dos críticos TD3 en cada ciclo acoplado.")
    g_rl.add_argument("--use_reward_centering", action=argparse.BooleanOptionalAction, default=True, help="Activa centrado adaptativo de recompensa (Reward Centering, Alg. 2 de Wang et al. 2025) en TD3.")
    g_rl.add_argument("--reward_centering_lr", type=float, default=0.01, help="Tasa de aprendizaje (EMA) para la media móvil de recompensa en Reward Centering.")
    g_rl.add_argument("--cdlb_exact_entropy", action=argparse.BooleanOptionalAction, default=False, help="Si es True incluye la constante exacta +(N/2)*log(2*pi) en r_explore=-alpha*log_q_fwd de CDLB, eliminando recompensas negativas para sigma<0.606 Bohr y el colapso alpha->0.")

    # 6. Entrenamiento
    g_trn = parser.add_argument_group("6. Régimen y Planificación de Entrenamiento")
    g_trn.add_argument("--train_mode", type=str, default="vmc_only", choices=["vmc_only", "rl_only", "coupled", "online_coupled", "eval_only"], help="Modo de entrenamiento a ejecutar.")
    g_trn.add_argument("--seed", type=int, default=42, help="Semilla pseudoaleatoria global.")
    g_trn.add_argument("--lr_ansatz", type=float, default=5e-4, help="Tasa de aprendizaje (Adam) para los parámetros del Ansatz.")
    g_trn.add_argument("--use_lr_scheduler", action=argparse.BooleanOptionalAction, default=True, help="Habilita o deshabilita el learning rate scheduler para todos los modelos entrenables (Ansatz VMC, Actor RL y Críticos).")
    g_trn.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "exponential", "linear", "constant"], help="Tipo de decaimiento del scheduler de learning rate.")
    g_trn.add_argument("--lr_warmup_steps", type=int, default=100, help="Pasos de calentamiento (warmup) del learning rate scheduler.")
    g_trn.add_argument("--lr_min_ratio", type=float, default=0.05, help="Cociente mínimo entre el learning rate final y el inicial (default: 0.05).")
    g_trn.add_argument("--grad_clip_norm", type=float, default=1.0, help="Norma global máxima para recorte de gradientes del Ansatz.")
    g_trn.add_argument("--train_steps", type=int, default=5000, help="Pasos totales de optimización si train_mode='vmc_only' o 'online_coupled'.")
    g_trn.add_argument("--rl_steps", "--vmc_steps", "--coupled_steps", dest="train_steps", type=int, help="Alias de --train_steps.")
    g_trn.add_argument("--coupled_training_cycles", type=int, default=8, help="Número de macro-ciclos alternando VMC y RL si train_mode='coupled'.")
    g_trn.add_argument("--vmc_iterations_per_cycle", type=int, default=1250, help="Pasos de optimización VMC por cada macro-ciclo acoplado.")
    g_trn.add_argument("--rl_sweeps_per_cycle", type=int, default=100, help="Barridos de entrenamiento RL por cada macro-ciclo acoplado (o totales en rl_only).")
    g_trn.add_argument("--online_nodal_warmup_steps", type=int, default=250, help="Fase 1 (Pipeline 4 online_coupled): pasos iniciales VMC con MALA sin guardar en Replay Buffer.")
    g_trn.add_argument("--online_critic_warmup_steps", type=int, default=250, help="Fase 2 (Pipeline 4 online_coupled): pasos VMC con MALA pre-llenando Replay Buffer (a=0) y entrenando solo Críticos TD3.")
    g_trn.add_argument("--online_rl_freeze_tail_steps", type=int, default=500, help="Fase 4 (Pipeline 4 online_coupled): pasos finales con Actor congelado (Diminishing Adaptation) para pulido variacional.")
    g_trn.add_argument("--online_rl_update_every", type=int, default=1, help="Frecuencia de pasos VMC entre actualizaciones de gradiente TD3 en modo online_coupled.")
    g_trn.add_argument("--online_rl_updates_per_step", type=int, default=1, help="Número de actualizaciones de gradiente TD3 por cada activación RL (ratio UTD) en modo online_coupled.")
    g_trn.add_argument("--online_walkers_to_store", type=int, default=64, help="Número de caminantes aleatorios por paso cuyas transiciones se guardan en el Replay Buffer en modo online_coupled (<=0 guarda todos).")
    g_trn.add_argument("--online_exp_noise", type=float, default=0.0, help="Ruido de exploración gaussiano en acciones durante la Fase 3 de online_coupled (0.0 preserva balance detallado exacto).")
    g_trn.add_argument("--online_reset_critic_period", type=int, default=0, help="Si > 0, reinicia alternadamente uno de los dos críticos TD3 cada N pasos durante la Fase 3 de online_coupled.")
    g_trn.add_argument("--steps_for_best", type=int, default=500, help="Pasos iniciales ignorados antes de registrar el mejor ansatz (menor energía).")
    g_trn.add_argument("--energy_clip_min", type=float, default=-500.0, help="Cota inferior dura para la energía local durante VMC.")
    g_trn.add_argument("--energy_clip_max", type=float, default=500.0, help="Cota superior dura para la energía local durante VMC.")
    g_trn.add_argument("--sigma_energy_clip", type=float, default=5.0, help="Múltiplo de MAD alrededor de la mediana para recortar outliers de energía local.")
    g_trn.add_argument("--n_print", type=int, default=100, help="Frecuencia de pasos para imprimir métricas en consola.")
    g_trn.add_argument("--n_log", type=int, default=50, help="Frecuencia de pasos para registrar métricas en el historial.")

    # 7. Evaluación / Producción
    g_evl = parser.add_argument_group("7. Fase de Evaluación / Producción MCMC (Ansatz y Actor Congelados)")
    g_evl.add_argument("--run_evaluation", action=argparse.BooleanOptionalAction, default=True, help="Ejecuta la cadena de producción al finalizar el entrenamiento.")
    g_evl.add_argument("--n_eval_chains", type=int, default=1, help="Número de cadenas independientes de evaluación a correr.")
    g_evl.add_argument("--eval_burn_in", type=int, default=2000, help="Pasos de termalización (burn-in) descartados en cada cadena de evaluación.")
    g_evl.add_argument("--eval_init_mode", type=str, default="same_as_sampler", choices=["same_as_sampler", "symmetric_gaussian", "alternating_spin", "spin_ordered_offset", "randomized_wigner", "uniform"], help="Modo de inicialización de caminantes al comenzar cada cadena de evaluación.")
    g_evl.add_argument("--measure_steps", type=int, default=5000, help="Pasos de producción recolectados en cada cadena de evaluación.")
    g_evl.add_argument("--thin", type=int, default=1, help="Factor de submuestreo (thinning) dentro de la GPU durante la evaluación.")
    g_evl.add_argument("--jit_chunk_size", type=int, default=1000, help="Tamaño de bloque lax.scan compilado para no saturar la memoria de GPU.")
    g_evl.add_argument("--eloc_chunk_size", type=int, default=20000, help="Tamaño de bloque para evaluar energías locales vectorizadas.")
    g_evl.add_argument("--use_actor_in_eval", action=argparse.BooleanOptionalAction, default=True, help="Mantiene encendido el Actor entrenado (sin ruido) durante la fase de evaluación.")

    # 8. Banderas de Guardado e I/O
    g_io = parser.add_argument_group("8. Banderas de Guardado y Archivo Unificado (.h5)")
    g_io.add_argument("--output_h5", type=str, default="results/run_bundle.h5", help="Ruta del archivo unificado HDF5 de salida.")
    g_io.add_argument("--results_out_dir", type=str, default=None, help="Si se especifica, guarda run_bundle.h5 (y opcionalmente archivos legacy) dentro de esta carpeta.")
    g_io.add_argument("--run_name", type=str, default=None, help="Nombre opcional del run para identificarlo en summary.json y en logs.")
    g_io.add_argument("--load_bundle_h5", type=str, default=None, help="Carga pesos de ansatz/actor desde un archivo unificado .h5 previo.")
    g_io.add_argument("--load_legacy_dir", type=str, default=None, help="Carga pesos de ansatz/actor desde una carpeta legacy previa (con archivos .pkl).")
    g_io.add_argument("--save_config", action=argparse.BooleanOptionalAction, default=True, help="Guarda la configuración completa de hiperparámetros en el archivo .h5.")
    g_io.add_argument("--save_ansatz_weights", action=argparse.BooleanOptionalAction, default=True, help="Guarda los pesos entrenados del Ansatz en el archivo .h5.")
    g_io.add_argument("--save_actor_weights", action=argparse.BooleanOptionalAction, default=True, help="Guarda los pesos entrenados del Actor neuronal en el archivo .h5.")
    g_io.add_argument("--save_critic_weights", action=argparse.BooleanOptionalAction, default=True, help="Guarda los pesos de las redes Críticas TD3 en el archivo .h5.")
    g_io.add_argument("--save_training_history", action=argparse.BooleanOptionalAction, default=True, help="Guarda las tablas de métricas de entrenamiento en el archivo .h5.")
    g_io.add_argument("--save_production_walkers", action=argparse.BooleanOptionalAction, default=True, help="Guarda las trayectorias de caminantes de la fase de producción en el archivo .h5.")
    g_io.add_argument("--save_walkers_max_batch", type=int, default=64, help="Número máximo de caminantes del lote a almacenar en /production/walkers (usar -1 para guardar todos).")
    g_io.add_argument("--save_production_energies", action=argparse.BooleanOptionalAction, default=True, help="Guarda las energías locales (total, cinética, potencial) de producción en el archivo .h5.")
    g_io.add_argument("--save_production_energy_mode", type=str, default="step_mean", choices=["step_mean", "full"], help="Modo de guardado de energías en /production: 'step_mean' guarda promedios por paso (n_chains, n_steps, 1) + atributos exactos de varianza y n_walkers (~128x más compacto); 'full' guarda todos los caminantes.")
    g_io.add_argument("--save_production_walkers_dtype", type=str, default="float32", choices=["float32", "float64"], help="Precisión numérica de /production/walkers en .h5 ('float32' reduce a la mitad el tamaño sin afectar histogramas ni D_KL).")
    g_io.add_argument("--save_production_stats_dtype", type=str, default="float32", choices=["float32", "float64"], help="Precisión numérica de /production/sampler_stats en .h5.")
    g_io.add_argument("--save_production_sampler_stats", action=argparse.BooleanOptionalAction, default=True, help="Guarda estadísticas de aceptación, desplazamientos y acciones de producción en el archivo .h5.")
    g_io.add_argument("--export_legacy_files", action=argparse.BooleanOptionalAction, default=False, help="Exporta adicionalmente params.json, *.pkl y *.csv en el formato antiguo.")
    g_io.add_argument("--save_summary_json", action=argparse.BooleanOptionalAction, default=True, help="Guarda summary.json en la carpeta del modelo con datos del sistema, hiperparámetros y observables finales.")
    g_io.add_argument("--master_csv", dest="master_csv_path", type=str, default=None, help="Ruta a tabla CSV incremental donde se anexa una fila en vivo al finalizar la corrida.")
    g_io.add_argument("--csv_collect_dir", type=str, default="results_csv", help="Carpeta central ligera donde se copian todos los archivos CSV generados con referencia al sistema.")
    g_io.add_argument("--reference_casscf_npz", type=str, default=None, help="Ruta explícita a casscf_state.npz de referencia para calcular ΔE y D_KL en summary.json.")
    g_io.add_argument("--uhf_search_root", type=str, default=".", help="Directorio raíz donde buscar carpetas uhf* con baselines CASSCF.")
    g_io.add_argument("--pipeline_stage", type=str, default="general", help="Identificador de etapa del pipeline (ej. 'pipeline2_sampling' o 'pipeline3_training').")

    return parser


def _sanitize_cli_tokens(tokens: list[str]) -> list[str]:
    """Sanitiza tokens CLI para evitar desajustes posicionales en flags booleanos y alias."""
    bool_flags = {
        # Experiment argparser
        "casscf_trainable",
        "use_spatial_gating",
        "legacy_drift_factor",
        "parity_equivariant",
        "unclip_sigma_action",
        "staggered_critic_reset",
        "use_reward_centering",
        "cdlb_exact_entropy",
        "use_lr_scheduler",
        "run_evaluation",
        "use_actor_in_eval",
        "save_config",
        "save_ansatz_weights",
        "save_actor_weights",
        "save_critic_weights",
        "save_training_history",
        "save_production_walkers",
        "save_production_energies",
        "save_production_sampler_stats",
        "export_legacy_files",
        "save_summary_json",
        # Postprocess argparser
        "auto_casscf_baseline",
        "recompute_local_energy",
        "compute_observables",
        "compute_diagnostics",
        "save_summary_csv",
        "save_observables_csv",
        "plot_training_metrics",
        "plot_production_summary",
        "plot_observables_comparison",
        "plot_efficiency_bars",
        "plot_time_to_solution",
        "plot_walker_trace",
        "plot_kl_divergence",
        "plot_hyperparameter_evolution",
    }
    sanitized: list[str] = []
    i = 0
    while i < len(tokens):
        t = str(tokens[i])

        # Manejo de formato --flag=value o --no-flag=value
        if "=" in t and t.startswith("--"):
            key, val = t.split("=", 1)
            val_clean = val.strip().lower()
            base_name = key[2:]
            is_negated = base_name.startswith("no-")
            clean_name = base_name[3:] if is_negated else base_name
            if clean_name in bool_flags:
                if val_clean in ("false", "0", "no", "f", "n"):
                    sanitized.append(f"--no-{clean_name}" if not is_negated else f"--{clean_name}")
                elif val_clean in ("true", "1", "yes", "t", "y"):
                    sanitized.append(f"--{clean_name}" if not is_negated else f"--no-{clean_name}")
                else:
                    sanitized.append(t)
                i += 1
                continue

        # Manejo de formato --flag value o --no-flag value
        if t.startswith("--"):
            base_name = t[2:]
            is_negated = base_name.startswith("no-")
            clean_name = base_name[3:] if is_negated else base_name
            if clean_name in bool_flags and (i + 1) < len(tokens):
                next_t = str(tokens[i + 1]).strip().lower()
                if next_t in ("false", "0", "no", "f", "n"):
                    sanitized.append(f"--no-{clean_name}" if not is_negated else f"--{clean_name}")
                    i += 2
                    continue
                elif next_t in ("true", "1", "yes", "t", "y"):
                    sanitized.append(f"--{clean_name}" if not is_negated else f"--no-{clean_name}")
                    i += 2
                    continue

        if t in ("--reward_type", "--reward_mode") and (i + 1) < len(tokens):
            if str(tokens[i + 1]).strip().lower() == "displacement":
                sanitized.append("--reward_type")
                sanitized.append("log_esjd")
                i += 2
                continue

        sanitized.append(t)
        i += 1
    return sanitized


def parse_experiment_args(argv: Optional[list[str]] = None) -> Experiment1DConfig:
    import sys
    raw_tokens = sys.argv[1:] if argv is None else list(argv)
    tokens = _sanitize_cli_tokens(raw_tokens)
    parser = build_experiment_argparser()
    args = parser.parse_args(tokens)
    if getattr(args, "reward_type", None) == "displacement":
        args.reward_type = "log_esjd"

    # Resolver atajo --method si fue proporcionado
    proposal_type = args.proposal_type
    train_mode = args.train_mode
    rl_algorithm = args.rl_algorithm
    if args.method is not None:
        m = args.method.upper()
        if m == "MH":
            proposal_type = "metropolis"
            if train_mode == "coupled":
                train_mode = "vmc_only"
        elif m == "MALA":
            proposal_type = "mala"
            if train_mode == "coupled":
                train_mode = "vmc_only"
        elif m == "HMC":
            proposal_type = "hmc"
            if train_mode == "coupled":
                train_mode = "vmc_only"
        elif m == "TD3":
            if proposal_type in ("metropolis", "mala", "hmc"):
                proposal_type = "neural_deepset"
            rl_algorithm = "td3"
            if args.train_mode == "vmc_only":
                train_mode = "coupled"
        elif m == "REINFORCE":
            if proposal_type in ("metropolis", "mala", "hmc"):
                proposal_type = "neural_deepset"
            rl_algorithm = "reinforce"
            if args.train_mode == "vmc_only":
                train_mode = "coupled"

    output_h5 = args.output_h5
    if args.results_out_dir is not None and args.output_h5 == "results/run_bundle.h5":
        import os
        output_h5 = os.path.join(args.results_out_dir, "run_bundle.h5")

    return Experiment1DConfig(
        system=System1DConfig(
            n_elec=args.n_elec,
            n_up=args.n_up,
            n_dn=args.n_dn,
            omega=args.omega,
            interaction_strength=args.interaction_strength,
            softening=args.softening,
        ),
        ansatz=Ansatz1DConfig(
            ansatz_type=args.ansatz_type,
            wf_model=args.wf_model,
            orbital_model=args.orbital_model,
            correlation_model=args.correlation_model,
            backflow_components=args.backflow_components,
            backflow_r_max=args.backflow_r_max,
            backflow_width=args.backflow_width,
            backflow_envelope_length=args.backflow_envelope_length,
            casscf_npz_path=args.casscf_npz_path,
            casscf_trainable=args.casscf_trainable,
        ),
        proposal=Proposal1DConfig(
            proposal_type=proposal_type,
            proposal_mode=args.proposal_mode,
            tau=args.tau,
            rw_sigma=args.rw_sigma,
            hmc_leapfrog_steps=args.hmc_leapfrog_steps,
            hmc_step_size=args.hmc_step_size,
            actor_amplitude=args.actor_amplitude,
            sigma_action_scale=args.sigma_action_scale,
            max_drift_clip=args.max_drift_clip,
            action_rev_limit=args.action_rev_limit,
            scale_clip_min=args.scale_clip_min,
            scale_clip_max=args.scale_clip_max,
            force_scale=args.force_scale,
            n_scale=args.n_scale,
            r_scale=args.r_scale,
            use_spatial_gating=args.use_spatial_gating,
            spatial_gating_radius=args.spatial_gating_radius,
            legacy_drift_factor=args.legacy_drift_factor,
            parity_equivariant=args.parity_equivariant,
            unclip_sigma_action=args.unclip_sigma_action,
        ),
        sampler=Sampler1DConfig(
            batch_size=args.batch_size,
            update_mode=args.update_mode,
            sweeps_per_step=args.sweeps_per_step,
            n_burn_in=args.n_burn_in,
            init_mode=args.init_mode,
            init_std=args.init_std,
            init_offset_base=args.init_offset_base,
            init_offset_step=args.init_offset_step,
            init_uniform_halfwidth=getattr(args, "init_uniform_halfwidth", 0.0),
            min_clip_grads_ansatz=args.min_clip_grads_ansatz,
            max_clip_grads_ansatz=args.max_clip_grads_ansatz,
            min_clip_log_ratio=args.min_clip_log_ratio,
            max_clip_log_ratio=args.max_clip_log_ratio,
            nan_clip_log_ratio=args.nan_clip_log_ratio,
            posinf_clip_log_ratio=args.posinf_clip_log_ratio,
            neginf_clip_log_ratio=args.neginf_clip_log_ratio,
        ),
        rl=RL1DConfig(
            algorithm=rl_algorithm,
            reward_type=args.reward_type,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            rl_gamma_td3=args.rl_gamma_td3,
            tau_polyak=args.tau_polyak,
            policy_delay=args.policy_delay,
            target_noise=args.target_noise,
            noise_clip=args.noise_clip,
            exp_noise_init=args.exp_noise_init,
            exp_noise_final=args.exp_noise_final,
            rl_replay_capacity=args.rl_replay_capacity,
            rl_batch_size=args.rl_batch_size,
            penalty_factor=args.penalty_factor,
            beta=args.beta,
            gamma=args.gamma,
            kappa=args.kappa,
            max_phys_jump_factor=args.max_phys_jump_factor,
            staggered_critic_reset=args.staggered_critic_reset,
            use_reward_centering=args.use_reward_centering,
            reward_centering_lr=args.reward_centering_lr,
            cdlb_exact_entropy=args.cdlb_exact_entropy,
            use_lr_scheduler=args.use_lr_scheduler,
            lr_scheduler_type=args.lr_scheduler_type,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_min_ratio=args.lr_min_ratio,
        ),
        train=Train1DConfig(
            mode=train_mode,
            seed=args.seed,
            lr_ansatz=args.lr_ansatz,
            use_lr_scheduler=args.use_lr_scheduler,
            lr_scheduler_type=args.lr_scheduler_type,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_min_ratio=args.lr_min_ratio,
            grad_clip_norm=args.grad_clip_norm,
            train_steps=args.train_steps,
            coupled_training_cycles=args.coupled_training_cycles,
            vmc_iterations_per_cycle=args.vmc_iterations_per_cycle,
            rl_sweeps_per_cycle=args.rl_sweeps_per_cycle,
            online_nodal_warmup_steps=args.online_nodal_warmup_steps,
            online_critic_warmup_steps=args.online_critic_warmup_steps,
            online_rl_freeze_tail_steps=args.online_rl_freeze_tail_steps,
            online_rl_update_every=args.online_rl_update_every,
            online_rl_updates_per_step=args.online_rl_updates_per_step,
            online_walkers_to_store=args.online_walkers_to_store,
            online_exp_noise=args.online_exp_noise,
            online_reset_critic_period=args.online_reset_critic_period,
            steps_for_best=args.steps_for_best,
            energy_clip_min=args.energy_clip_min,
            energy_clip_max=args.energy_clip_max,
            sigma_energy_clip=args.sigma_energy_clip,
            n_print=args.n_print,
            n_log=args.n_log,
        ),
        eval=Eval1DConfig(
            run_evaluation=args.run_evaluation,
            n_eval_chains=args.n_eval_chains,
            eval_burn_in=args.eval_burn_in,
            eval_init_mode=args.eval_init_mode,
            measure_steps=args.measure_steps,
            thin=args.thin,
            jit_chunk_size=args.jit_chunk_size,
            eloc_chunk_size=args.eloc_chunk_size,
            use_actor_in_eval=args.use_actor_in_eval,
        ),
        io=IOConfig(
            output_h5=output_h5,
            results_out_dir=args.results_out_dir,
            run_name=args.run_name,
            load_bundle_h5=args.load_bundle_h5,
            load_legacy_dir=args.load_legacy_dir,
            save_config=args.save_config,
            save_ansatz_weights=args.save_ansatz_weights,
            save_actor_weights=args.save_actor_weights,
            save_critic_weights=args.save_critic_weights,
            save_training_history=args.save_training_history,
            save_production_walkers=args.save_production_walkers,
            save_walkers_max_batch=args.save_walkers_max_batch,
            save_production_energies=args.save_production_energies,
            save_production_energy_mode=args.save_production_energy_mode,
            save_production_walkers_dtype=args.save_production_walkers_dtype,
            save_production_stats_dtype=args.save_production_stats_dtype,
            save_production_sampler_stats=args.save_production_sampler_stats,
            export_legacy_files=args.export_legacy_files,
            save_summary_json=args.save_summary_json,
            master_csv_path=args.master_csv_path,
            csv_collect_dir=args.csv_collect_dir,
            reference_casscf_npz=args.reference_casscf_npz,
            uhf_search_root=args.uhf_search_root,
            pipeline_stage=args.pipeline_stage,
        ),
    )


def build_postprocess_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RL-VMC Post-Procesamiento Modular: recibe uno o varios archivos unificados (.h5) y calcula diagnósticos, observables y figuras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_in = parser.add_argument_group("1. Archivos de Entrada y Salida")
    g_in.add_argument("--inputs", nargs="+", required=True, help="Uno o más archivos unificados .h5 (ej. mala.h5 td3.h5).")
    g_in.add_argument("--labels", nargs="+", default=None, help="Etiquetas personalizadas para cada archivo .h5 en las tablas y gráficas.")
    g_in.add_argument("--out_dir", type=str, default="results/postprocess", help="Directorio donde se guardarán las tablas y figuras.")
    g_in.add_argument("--reference_density_csv", type=str, default=None, help="CSV opcional con densidad exacta de referencia (FCI/CASSCF) para calcular D_KL.")
    g_in.add_argument("--casscf_npz_path", type=str, default=None, help="Ruta explícita opcional a un archivo casscf_state.npz de referencia.")
    g_in.add_argument("--uhf_search_root", type=str, default=".", help="Directorio raíz donde buscar carpetas uhf* con baselines CASSCF pre-entrenadas.")
    g_in.add_argument("--auto_casscf_baseline", action=argparse.BooleanOptionalAction, default=True, help="Si existe una baseline CASSCF pre-entrenada para el sistema físico (N, omega, g), superponerla automáticamente en las gráficas.")
    g_in.add_argument("--master_comparison_csv", type=str, default=None, help="Ruta a tabla CSV incremental donde se anexan las filas comparativas del sistema (incluyendo G_RL_speedup).")
    g_in.add_argument("--csv_collect_dir", type=str, default="results_csv", help="Carpeta central ligera donde se copian todos los archivos CSV generados con referencia al sistema.")
    g_in.add_argument("--system_tag", type=str, default=None, help="Etiqueta del sistema físico (ej. 'P2_N4_w0.1_g1.0') para nombrar archivos individuales.")

    g_calc = parser.add_argument_group("2. Parámetros de Cálculo de Observables y Diagnósticos")
    g_calc.add_argument("--recompute_local_energy", action=argparse.BooleanOptionalAction, default=False, help="Recalcula la energía local a partir de los caminantes y el ansatz guardado en el .h5.")
    g_calc.add_argument("--sokal_window_c", type=float, default=5.0, help="Constante de ventana automática de Sokal para integrar el tiempo de autocorrelación tau_int.")
    g_calc.add_argument("--max_acf_lag", type=int, default=2000, help="Lag máximo al estimar la función de autocorrelación.")
    g_calc.add_argument("--target_se_epsilon", type=float, default=1e-3, help="Error estándar objetivo (Hartree) para estimar el Time-to-Solution (T_epsilon).")
    g_calc.add_argument("--n_density_bins", type=int, default=120, help="Número de bins para histogramas de densidad de 1 cuerpo y pares.")
    g_calc.add_argument("--density_x_max", type=float, default=15.0, help="Rango espacial [-x_max, x_max] (Bohr) para evaluar la densidad de 1 cuerpo.")

    g_flags = parser.add_argument_group("3. Banderas de Cómputo y Guardado de Tablas")
    g_flags.add_argument("--compute_observables", action=argparse.BooleanOptionalAction, default=True, help="Calcula densidades de 1 cuerpo, distribución de pares P(r_ij) y observables físicos.")
    g_flags.add_argument("--compute_diagnostics", action=argparse.BooleanOptionalAction, default=True, help="Calcula ACF, tau_int, ESS, ESS/s, ESJD, MSD y Z-score de energía.")
    g_flags.add_argument("--save_summary_csv", action=argparse.BooleanOptionalAction, default=True, help="Guarda la tabla comparativa summary_metrics.csv y summary_metrics.json.")
    g_flags.add_argument("--save_observables_csv", action=argparse.BooleanOptionalAction, default=True, help="Guarda los perfiles espaciales de densidad y correlación de pares en archivos CSV.")

    g_plot = parser.add_argument_group("4. Banderas de Generación de Figuras")
    g_plot.add_argument("--plot_training_metrics", action=argparse.BooleanOptionalAction, default=True, help="Genera el dashboard comparativo de curvas de entrenamiento.")
    g_plot.add_argument("--plot_production_summary", action=argparse.BooleanOptionalAction, default=True, help="Genera la Figura 2 (Traza de E_L, ACF, Histograma de E_L y Error Estándar acumulado).")
    g_plot.add_argument("--plot_observables_comparison", action=argparse.BooleanOptionalAction, default=True, help="Genera la figura comparativa de densidad rho(x), pares P(r_ij) y Z-score.")
    g_plot.add_argument("--plot_efficiency_bars", action=argparse.BooleanOptionalAction, default=True, help="Genera la gráfica de barras de eficiencia (ESS/s y tau_int).")
    g_plot.add_argument("--plot_time_to_solution", action=argparse.BooleanOptionalAction, default=True, help="Genera la curva de Error Estándar SE(E) vs Tiempo de reloj (s).")
    g_plot.add_argument("--plot_walker_trace", action=argparse.BooleanOptionalAction, default=True, help="Genera la gráfica de trayectoria espacial (tunelamiento nodal) y movilidad MSD.")
    g_plot.add_argument("--plot_kl_divergence", action=argparse.BooleanOptionalAction, default=False, help="Genera la curva de convergencia de Divergencia KL respecto a reference_density_csv.")
    g_plot.add_argument("--plot_hyperparameter_evolution", action=argparse.BooleanOptionalAction, default=True, help="Genera las gráficas de evolución de precisión, D_KL y tau_int respecto a los hiperparámetros con barras de error.")
    g_plot.add_argument("--top_k_models", type=int, default=2, help="Número de mejores modelos de cada familia (ej. 2 MALA vs 2 RL-MALA) a graficar en las comparativas detalladas. Usar 0 o -1 para graficar todos.")
    g_plot.add_argument("--walker_trace_top_k", type=int, default=1, help="Número de mejores modelos por familia para la traza de un solo caminante (1 = mejor MALA vs mejor RL-MALA).")
    g_plot.add_argument("--n_eval_subblocks", type=int, default=5, help="Número de bloques/sub-evaluaciones independientes sobre los caminantes para calcular barras de error en D_KL, energía y tau_int.")
    g_plot.add_argument("--figure_format", "--format", dest="figure_format", type=str, default="png", choices=["pdf", "png", "both"], help="Formato de exportación para las figuras generadas ('png' por defecto para ahorrar espacio; usar 'pdf' o 'both' para artículo).")
    g_plot.add_argument("--dpi", "--ppi", dest="dpi", type=int, default=300, help="Resolución en píxeles/puntos por pulgada (PPI / DPI) de las figuras exportadas.")
    g_plot.add_argument("--title_prefix", type=str, default="", help="Prefijo o etiqueta personalizada para anteponer en los títulos de las figuras.")
    g_plot.add_argument("--custom_suptitle", type=str, default="", help="Título global personalizado (suptitle) para figuras de artículo.")
    g_plot.add_argument("--no_titles", "--hide_titles", dest="no_titles", action=argparse.BooleanOptionalAction, default=False, help="Oculta los títulos superiores de las figuras (útil para insertar en LaTeX con \\caption).")
    g_plot.add_argument("--font_scale", type=float, default=1.0, help="Factor de escala multiplicativo para el tamaño de todas las fuentes (ej. 1.2 para artículo).")
    g_plot.add_argument("--line_width_scale", type=float, default=1.0, help="Factor de escala multiplicativo para el grosor de las líneas.")
    g_plot.add_argument("--transparent_bg", action=argparse.BooleanOptionalAction, default=False, help="Exporta las figuras con fondo transparente.")

    return parser


def parse_postprocess_args(argv: Optional[list[str]] = None) -> PostProcessConfig:
    import sys
    raw_tokens = sys.argv[1:] if argv is None else list(argv)
    tokens = _sanitize_cli_tokens(raw_tokens)
    parser = build_postprocess_argparser()
    args = parser.parse_args(tokens)
    return PostProcessConfig(**vars(args))
