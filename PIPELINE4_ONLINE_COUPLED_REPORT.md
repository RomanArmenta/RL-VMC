# Pipeline 4 (`online_coupled`): Entrenamiento Variacional Acoplado Paso a Paso en Dos Escalas de Tiempo y Cuatro Regímenes

**Autor / Framework:** `rl_vmc` — Deep Deterministic Quantum Monte Carlo (1D Interacting Quantum Dots & Wigner Molecules)  
**Fecha:** Noviembre 2026  
**Módulos Principales:** [`rl_vmc/trainers/online_coupled_trainer_1d.py`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/trainers/online_coupled_trainer_1d.py), [`rl_vmc/rl/learners_1d.py`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/rl/learners_1d.py), [`rl_vmc/trainers/runner_1d.py`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/trainers/runner_1d.py), [`rl_vmc/config.py`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/config.py)  
**Orquestadores de Producción y Búsqueda:** [`run_pipeline4_param_search.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_pipeline4_param_search.sh), [`run_pipeline4_benchmark.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_pipeline4_benchmark.sh), [`run_production_benchmark_3pipelines.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_production_benchmark_3pipelines.sh)

---

## 1. Resumen Ejecutivo y Diagnóstico Computacional

En el **Pipeline 3 (`coupled`)** original, el entrenamiento variacional de la función de onda $\Psi_\theta(X)$ (tipo Slater-Jastrow-Backflow o Multi-Determinante SJB) y el entrenamiento del sampler neuronal **RL-MALA** (Actor DeepSet + Críticos gemelos TD3) se realizaban en **macro-ciclos alternados y desconectados**:

1. Durante `_run_vmc_block` ($1,200$ pasos VMC $\times 5$ sweeps/paso $= 6,000$ pasos MCMC por ciclo), el método [`MHSampler1D.step`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/samplers/mh_sampler_1d.py#L125-L230) calculaba en cada barrido el diccionario completo de transición de Reinforcement Learning (`rl_info`: características de estado $s_t$, acción $a_t$, siguiente estado $s_{t+1}$, probabilidad de aceptación $\alpha_t$, diferencia de log-probabilidad $\Delta \log |\Psi_\theta|^2$, densidad de propuesta $\log q_{\phi}(X' \mid X)$ y salto cuadrático $\|X' - X\|^2$), **pero `_run_vmc_block` lo descartaba con `state, stats, _ = sampler.step(...)`**.
2. A continuación, `_run_rl_block` congelaba $\Psi_\theta$ y ejecutaba $400$ pasos adicionales de MCMC por ciclo ($4,400$ pasos extra en $10$ ciclos considerando el bloque de pulido final) exclusivamente para poblar el Replay Buffer y actualizar el Actor y los Críticos TD3, **sin actualizar $\theta$**.

El **Pipeline 4 (`online_coupled`)**, implementado en la nueva clase [`OnlineCoupledTrainer1D`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/trainers/online_coupled_trainer_1d.py), elimina por completo esa duplicidad de trabajo acoplando la optimización variacional (VMC) y el aprendizaje por refuerzo (TD3) **paso a paso dentro de un único bucle de evolución conjunta**, estructurado en **4 Regímenes físicamente motivados**:

1. **Régimen 1 — Relajación Nodal Rápida (`Nodal Warmup`, $1 \le t \le N_{\text{nodal}}$)**: Muestreo con MALA puro (`use_actor=False`, $a = (\mathbf{0}, \mathbf{0})$) y actualización variacional de $\Psi_\theta$ sin guardar aún en el Replay Buffer, evitando contaminar la memoria de experiencia con el transiente inicial donde la estructura nodal cambia abruptamente.
2. **Régimen 2 — Pre-llenado Off-Policy Gratuito y Calentamiento del Crítico (`Critic Warmup`, $N_{\text{nodal}} < t \le N_{\text{nodal}} + N_{\text{critic}}$)**: Continúa muestreando con MALA puro ($a = \mathbf{0}$) y actualizando $\Psi_\theta$, pero **recicla sin costo adicional el `rl_info` de cada paso VMC** para poblar el Replay Buffer con transiciones $(s_t, a_t = \mathbf{0}, r_t^{\text{CDLB}}, s_{t+1})$ y entrenar **únicamente las redes Críticas gemelas** $Q_{\vartheta_1}, Q_{\vartheta_2}$ (`allow_actor_update=False`). Así, antes de que el Actor dé su primer paso, los Críticos ya convergieron a la función de valor real de la política base MALA.
3. **Régimen 3 — Co-Evolución Acoplada en Dos Escalas de Tiempo (`Two-Timescale Co-Evolution`, $N_{\text{nodal}} + N_{\text{critic}} < t \le T_{\text{total}} - N_{\text{freeze}}$)**: Activa el Actor neuronal (`use_actor=True`) en modo determinista ($\varepsilon_{\text{expl}} = 0$ por defecto, preservando el balance detallado exacto en cada paso de Metropolis-Hastings), actualiza $\Psi_\theta$ con los caminantes muestreados y actualiza simultáneamente los Críticos y el Actor TD3 reutilizando las trazas generadas en ese mismo paso VMC.
4. **Régimen 4 — Adaptación Decreciente y Pulido Variacional con Kernel Congelado (`Tail Freeze / Diminishing Adaptation`, $T_{\text{total}} - N_{\text{freeze}} < t \le T_{\text{total}}$)**: Congela los pesos del Actor y Críticos ($\Delta \phi_t = 0$), manteniendo activo el sampler neuronal entrenado para finalizar la convergencia fina de $\Psi_\theta$ y seleccionar `best_params_psi` bajo un kernel de Markov estrictamente estacionario y ergódico.

---

## 2. Marco Teórico y Formulación Matemática Completa

### 2.1. Hamiltoniano Físico y Ansatz Variacional Slater-Jastrow-Backflow (SJB)

Consideramos $N = N_\uparrow + N_\downarrow$ electrones interactuantes en una trampa armónica unidimensional con repulsión de Coulomb suavizada (*soft-Coulomb*), implementado en [`Harmonic1DHamiltonian`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/physics/harmonic_1d.py):

$$\hat{H} = \sum_{i=1}^{N} \left( -\frac{1}{2}\frac{\partial^2}{\partial x_i^2} + \frac{1}{2}\omega^2 x_i^2 \right) + \sum_{1 \le i < j \le N} \frac{g}{\sqrt{(x_i - x_j)^2 + a^2}}$$

La función de onda variacional $\Psi_\theta(X)$ parametrizada por $\theta$ ([`SlaterJastrow1DAnsatz`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/ansatze/ansatz_1d.py) o [`MultiSlaterJastrowBackflow1DAnsatz`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/ansatze/ansatz_1d.py)) combina determinantes de Slater evaluados sobre coordenadas transformadas por **Backflow** $\tilde{x}_i(X)$ con un factor de correlación de **Jastrow-RBF** dependiente del espín $J_\theta(X)$:

$$\log |\Psi_\theta(X)| = \log \left| \sum_{k=1}^{M_{\text{det}}} c_k \det\!\left[\phi_{k,a}^\uparrow(\tilde{x}_{i,\uparrow})\right] \det\!\left[\phi_{k,b}^\downarrow(\tilde{x}_{j,\downarrow})\right] \right| + J_\theta(X)$$

donde las coordenadas cuasi-partícula de Backflow desplazan los nodos de acuerdo con el entorno inter-electrónico:

$$\tilde{x}_i = x_i + \sum_{j \ne i} \eta_{\sigma_i \sigma_j}(x_i - x_j), \qquad \eta_{\sigma_i \sigma_j}(r_{ij}) = r_{ij} \, e^{-(r_{ij}/L_{\text{env}})^2} \sum_{m=1}^{K_{\text{bf}}} w_{m,\sigma_i\sigma_j} \exp\!\left(-\frac{(|r_{ij}| - \mu_m)^2}{2w_{\text{bf}}^2}\right)$$

### 2.2. Energía Local y Gradiente Variacional Robusto con Recorte MAD (`vmc_update_step`)

Para un lote de $B$ caminantes $X^{(b)} \in \mathbb{R}^{N \times 1}$, la energía local $E_L(X) = \Psi_\theta(X)^{-1} \hat{H} \Psi_\theta(X)$ se evalúa exactamente en modo logarítmico mediante diferenciación automática en JAX ([`Harmonic1DHamiltonian.local_energy_and_components`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/physics/harmonic_1d.py)):

$$E_L(X) = -\frac{1}{2} \sum_{i=1}^{N} \left[ \frac{\partial^2 \log|\Psi_\theta(X)|}{\partial x_i^2} + \left(\frac{\partial \log|\Psi_\theta(X)|}{\partial x_i}\right)^2 \right] + \frac{1}{2}\omega^2 \sum_{i=1}^{N} x_i^2 + \sum_{i < j} \frac{g}{\sqrt{(x_i - x_j)^2 + a^2}}$$

Para evitar que singularidades cercanas a los nodos produzcan explosiones de varianza en el gradiente variacional, [`vmc_update_step`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/trainers/runner_1d.py#L245-L290) aplica recorte estadístico basado en la **Desviación Absoluta Mediana (MAD)**:

$$m_t = \text{median}_{b}\!\left(E_L(X_t^{(b)})\right), \qquad \text{MAD}_t = \text{median}_{b}\!\left(\left|E_L(X_t^{(b)}) - m_t\right|\right) + 10^{-8}$$

$$\tilde{E}_L(X_t^{(b)}) = \text{clip}\!\left(E_L(X_t^{(b)}),\; m_t - k_{\text{MAD}}\,\text{MAD}_t,\; m_t + k_{\text{MAD}}\,\text{MAD}_t\right), \qquad \bar{E}_t = \frac{1}{B}\sum_{b=1}^{B} \tilde{E}_L(X_t^{(b)})$$

El estimador del gradiente de la energía variacional $E(\theta) = \mathbb{E}_{X \sim |\Psi_\theta|^2}[E_L(X)]$ se obtiene diferenciando la pérdida sustituta (*surrogate loss*):

$$\mathcal{L}_{\text{VMC}}(\theta) = \frac{2}{B} \sum_{b=1}^{B} \text{stop\_gradient}\!\left(\tilde{E}_L(X_t^{(b)}) - \bar{E}_t\right) \log |\Psi_\theta(X_t^{(b)})|, \qquad \nabla_\theta E(\theta) = \nabla_\theta \mathcal{L}_{\text{VMC}}(\theta)$$

seguido de recorte por norma global (`optax.clip_by_global_norm`) y actualización con Adam (`optax.adam`).

### 2.3. Propuesta Neuronal Residual-MALA con DeepSets Equivariantes (`DeepSetNeuralProposal1D`)

En [`DeepSetNeuralProposal1D`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py#L219-L410), para cada electrón $i \in \{1, \dots, N\}$ de cada caminante, la función [`build_particle_features_1d`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py#L33-L111) construye el vector de estado equivariante $s_t^{(b,i)} \in \mathbb{R}^{d_s}$ compuesto por:

1. **Características locales ($4$ dims)**:
   $$f_{\text{loc}}^{(b,i)} = \left[\, x_i,\;\; x_i - \bar{x}^{(b)},\;\; \tanh\!\left(\frac{\nabla_{x_i}\log|\Psi_\theta(X^{(b)})|}{s_F}\right),\;\; \sigma_i \,\right], \qquad \sigma_i \in \{+1, -1\}$$
2. **Características globales del sistema ($5$ dims)**:
   $$f_{\text{glob}} = \left[\, \frac{N}{s_N},\;\; \frac{N_\uparrow}{N},\;\; \tanh(\omega),\;\; \tanh(g),\;\; \tanh(a) \,\right]$$
3. **Slots relacionales ordenados por distancia para vecinos del mismo espín y espín opuesto**:
   Para cada vecino $j \ne i$, se codifican las ternas $\big(|x_i - x_j|/s_r,\; \text{sign}(x_i - x_j),\; \mathbb{I}_{\text{activo}}\big)$ separadas en canales $\uparrow\uparrow / \downarrow\downarrow$ (nodo de Pauli) y $\uparrow\downarrow$ (cúspide de Coulomb / barrera de Wigner), las cuales se procesan mediante encoders MLP con activación SiLU y *sum-pooling* invariante por permutación en [`FlaxTD3Actor`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py#L114-L162).

El Actor neuronal $\pi_\phi: \mathbb{R}^{d_s} \to [-1, 1] \times [-s_\sigma, s_\sigma]$ produce dos acciones continuas por electrón $a_t^{(b,i)} = \big(a_{\mu}^{(b,i)},\, a_{\sigma}^{(b,i)}\big)$:

$$a_\mu^{(b,i)} = \tanh\!\big(z_\mu^{(b,i)}\big) \in (-1, 1), \qquad a_\sigma^{(b,i)} = s_\sigma \tanh\!\big(z_\sigma^{(b,i)}\big) \in (-s_\sigma, s_\sigma)$$

**Propiedad fundamental de inicialización cero**: En [`FlaxTD3Actor`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py#L151-L152), las capas lineales de salida `Dense(1)` de $z_\mu$ y $z_\sigma$ se inicializan idénticamente en cero (`kernel_init=nn.initializers.zeros`). Por lo tanto, en $t=0$ (y durante todo el Régimen 1 y Régimen 2), $\pi_{\phi_0}(s) \equiv (\mathbf{0}, \mathbf{0})$.

Bajo el modo `residual_mala` ([`ResidualMALAPolicy1D.compute_distribution`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py#L170-L216)), la propuesta gaussiana $X' \sim q_\phi(\cdot \mid X) = \mathcal{N}\!\big(\mu_\phi(X),\, \text{diag}(\sigma_\phi^2(X))\big)$ tiene media y desviación estándar:

$$\mu_{\phi,i}(X) = x_i + \underbrace{\tau \, \text{clip}\!\left(\nabla_{x_i}\log|\Psi_\theta(X)|, -D_{\max}, D_{\max}\right)}_{\text{Deriva MALA Clásica}} + \underbrace{A \, g_{\text{spat}}(x_i)\, a_{\mu,i}(s)}_{\text{Corrección Neuronal } \delta\mu_i}$$

$$\sigma_{\phi,i}(X) = \sqrt{2\tau} \, \exp\!\left( \text{clip}\!\left(g_{\text{spat}}(x_i)\, a_{\sigma,i}(s),\; c_{\min},\; c_{\max}\right) \right)$$

donde $A$ es `actor_amplitude`, $s_\sigma$ es `sigma_action_scale`, y $g_{\text{spat}}(x_i) = \big(1 + (x_i / R_{\text{gate}})^2\big)^{-1/2}$ es el factor de contención espacial ([`apply_spatial_gating`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py#L11-L30)). Cuando $a = (\mathbf{0}, \mathbf{0})$, $\mu_{\phi,i}(X) = x_i + \tau \nabla_{x_i}\log|\Psi_\theta(X)|$ y $\sigma_{\phi,i}(X) = \sqrt{2\tau}$, recuperando **exactamente el algoritmo MALA clásico**.

### 2.4. Filtro Exacto de Metropolis-Hastings e Invariancia Instantánea de $|\Psi_{\theta_t}|^2$

En [`MHSampler1D.step`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/samplers/mh_sampler_1d.py#L125-L230), dada la propuesta $X' = \mu_\phi(X) + \sigma_\phi(X) \odot \xi$ con $\xi \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, se evalúan $\log|\Psi_{\theta_t}(X')|$ y $\nabla_{X'}\log|\Psi_{\theta_t}(X')|$ mediante `ansatz.value_and_grad`, así como la distribución inversa $\mu_\phi(X'), \sigma_\phi(X')$ para calcular la probabilidad de aceptación de Metropolis-Hastings:

$$\log \alpha_t(X, X') = \min\!\left(0,\;\; \underbrace{2\big(\log|\Psi_{\theta_t}(X')| - \log|\Psi_{\theta_t}(X)|\big)}_{\Delta \log p_t} + \log q_{\phi_t}(X \mid X') - \log q_{\phi_t}(X' \mid X)\right)$$

Como demuestran Wang, Chen, Kanagawa & Oates (2024) en [*Reinforcement Learning for Adaptive MCMC* (`2405.13574v1-2.pdf`, Remark 3, pág. 7 y Teorema 2, pág. 9)](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/referencias/2405.13574v1-2.pdf):
- Si la cadena de Markov se evoluciona utilizando la política determinista $\pi_{\phi_t}(s)$ sin inyectar ruido estocástico externo en la acción ($\varepsilon_{\text{expl}} = 0$), el kernel de transición de Metropolis-Hastings $K_{\theta_t, \phi_t}(X, dX')$ satisface **balance detallado exacto** respecto a la densidad objetivo instantánea $p_{\theta_t}(X) \propto |\Psi_{\theta_t}(X)|^2$:
  $$p_{\theta_t}(X) \, K_{\theta_t, \phi_t}(X, dX') = p_{\theta_t}(X') \, K_{\theta_t, \phi_t}(X', dX)$$
- Al mismo tiempo, debido a que tanto los parámetros variacionales $\theta_t$ como los parámetros del Actor $\phi_t$ evolucionan paso a paso, los pares estado-acción $(s_t, a_t)$ que ingresan al Replay Buffer $\mathcal{D}$ poseen diversidad suficiente para entrenar fuera de política (*off-policy*) a los Críticos TD3 sin necesidad de corromper la cadena de VMC con ruido de exploración alto.

### 2.5. Recompensa CDLB (*Contrastive Divergence Lower Bound*) en Paisajes No-Estacionarios

En [`compute_cdlb_reward`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/rl/rewards.py#L31-L82), implementamos la recompensa **CDLB** propuesta por Wang, Fisher, Kanagawa, Chen & Oates (2025) en [*Harnessing the Power of Reinforcement Learning for Adaptive MCMC* (`2507.00671v1.pdf`, Secciones 2.2 y 2.3, págs. 7–8)](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/referencias/2507.00671v1.pdf).

A diferencia de ESJD, que asume que la cadena ya alcanzó la distribución estacionaria, CDLB maximiza una cota inferior variacional de la reducción en un paso de la divergencia de Kullback-Leibler entre la distribución actual de los caminantes $P_t$ y la densidad objetivo instantánea $P_{\theta_t} = |\Psi_{\theta_t}|^2 / Z_{\theta_t}$:

$$\Delta_t = D_{\text{KL}}\!\big(P_t \parallel P_{\theta_t}\big) - D_{\text{KL}}\!\big(P_{t+1} \parallel P_{\theta_t}\big)$$

La recompensa por caminante se descompone en tres contribuciones físicas complementarias (todas evaluadas directamente a partir de `rl_info` con **cero evaluaciones adicionales del ansatz**):

$$r_t^{\text{CDLB}}(X, X') = \lambda_{\text{pen}} \Big( r_{\text{exploit}}(X, X') + r_{\text{entropy}}(\alpha_t) \Big) + r_{\text{explore}}(X, X')$$

donde:
1. **Término de Explotación (ascenso hacia regiones de alta probabilidad $|\Psi_{\theta_t}|^2$)**:
   $$r_{\text{exploit}} = \text{clip}\!\Big(\alpha_t \cdot 2\big(\log|\Psi_{\theta_t}(X')| - \log|\Psi_{\theta_t}(X)|\big),\; -5.0,\; 5.0\Big)$$
2. **Término de Entropía Binaria de Aceptación-Rechazo (estabilización alrededor de tasas de aceptación óptimas $\alpha \in [0.3, 0.7]$)**:
   $$r_{\text{entropy}} = -\alpha_t \log \alpha_t - (1 - \alpha_t)\log(1 - \alpha_t)$$
3. **Término de Exploración Acotada (movilidad en el espacio de configuraciones y cruce de barreras nodales)**:
   $$r_{\text{explore}} = \alpha_t \log\!\left(1 + \frac{\min\!\big(\|X' - X\|_1 / N,\; d_{\max}\big)^2}{\epsilon_r}\right), \qquad d_{\max} = c_{\text{jump}}\sqrt{\tau}$$

### 2.6. Aprendizaje Off-Policy TD3 con *Reward Centering* y Dos Escalas de Tiempo

Como se analiza en Kevin P. Murphy (2025) [*Reinforcement Learning: A Comprehensive Overview* (`2412.05265v2.pdf`, Secciones 2.5.2.7, 3.3.3 y 4.3.1)](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/referencias/2412.05265v2.pdf) y en Yang et al. (2018) [*Mean Field Multi-Agent RL* (`1802.05438v5.pdf`, Teoremas 1 y 2)](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/referencias/1802.05438v5.pdf), la co-evolución de $\Psi_{\theta_t}$ y del sampler $(\phi_t, \vartheta_{1,t}, \vartheta_{2,t})$ constituye un sistema estocástico acoplado de dos escalas de tiempo.

En [`TD3Learner1D`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/rl/learners_1d.py#L75-L420), incorporamos tres mecanismos específicos para garantizar estabilidad cuando $\Psi_{\theta_t}$ evoluciona paso a paso:

1. **Centrado Adaptativo de Recompensa (*Reward Centering*, Algoritmo 2 de Wang et al., 2025 [`2507.00671v1.pdf`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/referencias/2507.00671v1.pdf), Apéndice D.1, págs. 21–22)**:
   Dado que el muestreo MCMC es una tarea continua no episódica y el nivel medio de recompensa cambia conforme $\Psi_{\theta_t}$ relaja sus nodos, mantenemos un estimador de media móvil exponencial $\bar{R}_t$:
   $$\bar{R}_{k} = (1 - \eta_R)\bar{R}_{k-1} + \eta_R \left(\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} r_i\right)$$
   y construimos el objetivo de Bellman centrado para las redes Críticas gemelas $Q_{\vartheta_1}, Q_{\vartheta_2}$:
   $$\tilde{a}'_i = \text{clip}\!\Big(\pi_{\phi'}(s'_i) + \text{clip}(\epsilon, -c, c),\; -1, 1\Big), \qquad \epsilon \sim \mathcal{N}(0, \sigma_{\text{target}}^2 \mathbf{I})$$
   $$y_i = \big(r_i - \bar{R}_k\big) + \gamma_{\text{TD3}} \min_{j=1,2} Q_{\vartheta'_j}\!\big(s'_i,\, \tilde{a}'_i\big)$$
   minimizando $\mathcal{L}_{\text{critic}}(\vartheta_j) = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \big(Q_{\vartheta_j}(s_i, a_i) - y_i\big)^2$ para $j \in \{1, 2\}$.

2. **Separación de Escalas de Tiempo y Actualización Diferida del Actor**:
   Los Críticos se actualizan con tasa $\eta_{\text{critic}} = 3 \times 10^{-4}$ en cada paso, mientras que el Actor se actualiza únicamente cada $d_{\text{delay}} = 2$ pasos (y solo cuando `allow_actor_update=True` en el Régimen 3) con tasa menor $\eta_{\text{actor}} = 1 \times 10^{-4}$ maximizando $Q_{\vartheta_1}$:
   $$\mathcal{L}_{\text{actor}}(\phi) = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} Q_{\vartheta_1}\!\big(s_i,\, \pi_\phi(s_i)\big)$$
   Las redes objetivo $(\phi', \vartheta'_1, \vartheta'_2)$ evolucionan suavemente mediante promediado de Polyak ($\tau_{\text{polyak}} = 0.005$):
   $$\vartheta'_j \leftarrow (1 - \tau_{\text{polyak}})\vartheta'_j + \tau_{\text{polyak}}\vartheta_j, \qquad \phi' \leftarrow (1 - \tau_{\text{polyak}})\phi' + \tau_{\text{polyak}}\phi$$

3. **Ventana FIFO Corta y Submuestreo de Caminantes (`online_walkers_to_store`)**:
   En lugar de insertar los $B = 512$ caminantes de un mismo instante $t$ (que saturarían el buffer con una sola versión de $\Psi_{\theta_t}$), [`TD3Learner1D.push_transitions`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/rl/learners_1d.py#L221-L246) selecciona aleatoriamente $M = 64$ caminantes ($64 \times N$ transiciones de partícula por paso) en un buffer circular FIFO de capacidad $C_{\text{replay}} \in [15\,000, 30\,000]$. Así, el buffer contiene siempre una ventana deslizante fresca de los últimos $\sim 60$–$120$ pasos VMC, descartando automáticamente transiciones evaluadas bajo funciones de onda obsoletas.

---

## 3. Anatomía Detallada de los 4 Regímenes del Pipeline 4 (`OnlineCoupledTrainer1D`)

La clase [`OnlineCoupledTrainer1D`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/trainers/online_coupled_trainer_1d.py) divide los $T_{\text{total}}$ pasos de entrenamiento en cuatro regímenes consecutivos definidos por los puntos de transición:
$$T_1 = N_{\text{nodal}}, \qquad T_2 = N_{\text{nodal}} + N_{\text{critic}}, \qquad T_3 = T_{\text{total}} - N_{\text{freeze}}$$

| Régimen | Rango de Pasos $t$ | Sampler Activo en MCMC | ¿Inserta en Replay Buffer $\mathcal{D}$? | ¿Actualiza Críticos $Q_{\vartheta_{1,2}}$? | ¿Actualiza Actor $\pi_\phi$? | ¿Actualiza Ansatz $\Psi_\theta$? | Propósito Físico y Estadístico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **R1: `NodalWarmup`** | $1 \le t \le T_1$ | MALA puro ($a=\mathbf{0}$) | No | No | No ($\phi = \phi_0 \equiv \mathbf{0}$) | **Sí** | Relajar el transiente inicial de nodos Backflow y cúspides Jastrow sin contaminar el buffer. |
| **R2: `CriticWarmup`** | $T_1 < t \le T_2$ | MALA puro ($a=\mathbf{0}$) | **Sí** (recicla `rl_info` gratis) | **Sí** (`allow_actor_update=False`) | No ($\phi = \phi_0 \equiv \mathbf{0}$) | **Sí** | Pre-llenar $\mathcal{D}$ gratis y converger $Q_{\vartheta}(s, a=\mathbf{0})$ al valor real de MALA antes de mover el Actor. |
| **R3: `CoEvolution`** | $T_2 < t \le T_3$ | **Actor Neuronal** (`use_actor=True`, $\varepsilon=0$) | **Sí** (recicla `rl_info` gratis) | **Sí** | **Sí** (cada `policy_delay` pasos) | **Sí** | Co-evolución paso a paso en dos escalas de tiempo usando las mismas muestras para VMC y TD3. |
| **R4: `TailFreeze`** | $T_3 < t \le T_{\text{total}}$ | **Actor Neuronal Congelado** ($\Delta\phi=0, \varepsilon=0$) | No | No | No (Congelado) | **Sí** (+ selección `best_params_psi`) | Garantizar *Diminishing Adaptation* ($\|\phi_{t+1}-\phi_t\|=0$, Teorema 2 de Wang et al. 2024) y ergodicidad exacta. |

### ¿Por qué el Régimen 2 (Pre-llenado Off-Policy + Calentamiento del Crítico) es la pieza clave?
Cuando el Actor [`FlaxTD3Actor`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py#L151-L152) está en su inicialización cero ($\phi = \phi_0$), su política es idénticamente $\pi_{\phi_0}(s) = (\mathbf{0}, \mathbf{0})$, la cual coincide **exactamente** con la propuesta de MALA clásico (`use_actor=False`).
- Si activáramos la actualización del Actor en cuanto el buffer tiene su primer minibatch (sin Régimen 2), el Crítico $Q_\vartheta(s, a)$ aún tendría pesos aleatorios, por lo que el gradiente determinista de política $\nabla_\phi \mathcal{L}_{\text{actor}} = -\mathbb{E}[\nabla_a Q_\vartheta(s, a)|_{a=0} \nabla_\phi \pi_\phi(s)]$ empujaría al Actor en una dirección aleatoria, degradando la aceptación MCMC en plena optimización de $\Psi_\theta$.
- En cambio, durante el **Régimen 2**, como muestreamos con MALA ($a = \mathbf{0}$) y entrenamos únicamente los Críticos ($Q_{\vartheta_1}, Q_{\vartheta_2}$) y sus redes target ($Q_{\vartheta'_1}, Q_{\vartheta'_2}$), las transiciones $(s_t, a_t=\mathbf{0}, r_t, s_{t+1})$ son **simultáneamente on-policy para $\pi_{\phi_0}$ y off-policy para TD3**. Al iniciar el Régimen 3 en $t = T_2 + 1$, el Crítico ya conoce la superficie de valor alrededor de $a = \mathbf{0}$ y el primer paso del Actor mejora monotónicamente sobre MALA.

---

## 4. Especificación Exacta de Clases y Funciones en el Código

1. **[`OnlineCoupledTrainer1D`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/trainers/online_coupled_trainer_1d.py#L22-L343)** (`rl_vmc/trainers/online_coupled_trainer_1d.py`):
   - **`__init__(self, cfg, ansatz, proposal, sampler, learner, vmc_update_fn, loaded_actor=False)`**: Calcula las fronteras exactas de los 4 regímenes (`step_end_phase1`, `step_end_phase2`, `step_end_phase3`) y configura el submuestreo al buffer y la frecuencia UTD.
   - **`get_phase_id_and_label(self, step: int) -> tuple[int, str]`**: Determina en tiempo constante a cuál de los 4 regímenes pertenece el paso actual $t$.
   - **`run(self, key, state, params_psi, vmc_opt_state, params_actor, best_params_psi, best_energy) -> dict`**: Ejecuta el bucle acoplado paso a paso, cronometrando el tiempo de cómputo crudo (`time.perf_counter()`) de cada paso y de cada régimen (`phase_wall_times_sec`), reutilizando el `rl_info_last` devuelto por `self.sampler.step(...)` para llamar a `compute_rl_reward(rl_info_last, self.cfg)`, `self.learner.push_transitions(...)`, `self.learner.update_step(k_u, allow_actor_update=(phase_id == 3))` y `self.vmc_update_fn(params_psi, vmc_opt_state, state.walkers)`.
2. **[`TD3Learner1D`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/rl/learners_1d.py#L75-L420)** (`rl_vmc/rl/learners_1d.py`):
   - **`push_transitions(self, rl_info, rewards, max_walkers=None)`**: Reestructura las transiciones por caminante $(B, N, d_s)$, aplica submuestreo aleatorio sin reemplazo de `max_walkers` caminantes cuando `max_walkers < B` y las inserta en [`ReplayBuffer.add_batch`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/rl/learners_1d.py#L35-L45).
   - **`update_step(self, key, allow_actor_update=True)`**: Muestrea un minibatch de tamaño `rl_batch_size`, actualiza la media móvil de recompensa $\bar{R}$ (`running_reward_mean`) si `use_reward_centering=True`, y ejecuta [`_train_networks`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/rl/learners_1d.py#L247-L366) compilado con `@jax.jit`. Cuando `allow_actor_update=False` (Régimen 2), congela el Actor pero mantiene activa la actualización de Polyak de los Críticos objetivo (`update_targets`).
3. **[`run_experiment_1d`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/trainers/runner_1d.py#L86-L1019)** (`rl_vmc/trainers/runner_1d.py`):
   - Imprime el encabezado completo con marcas de tiempo (`YYYY-MM-DD HH:MM:SS`) y el JSON íntegro de hiperparámetros en el log individual.
   - Despacha `cfg.train.mode == "online_coupled"` a `OnlineCoupledTrainer1D`.
   - Ejecuta la evaluación multi-cadena con pesos congelados (`sampler.run_production_chain`) y exporta `run_bundle.h5`, `summary.json`, `training_history_<run>.csv` y `production_metrics_<run>.csv` incluyendo `train_wall_time_sec`, `burn_in_wall_time_sec`, `wall_time_sec`, `total_run_wall_time_sec` y `phase_wall_times_sec`.
4. **[`run_postprocessing_pipeline`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/postprocess/plotter.py#L1921-L2415)** (`rl_vmc/postprocess/plotter.py`):
   - Además de generar las figuras en `.png` (o `.pdf` a alto DPI) y las tablas `summary_metrics_*.csv` y `observables_*.csv`, exporta automáticamente los CSVs crudos de cada figura (`figure_data_acf_energy_*.csv`, `figure_data_kl_convergence_*.csv`, `figure_data_walker_traces_*.csv`) en `results_csv/`.

---

## 5. Comparación Computacional: Pipeline 3 (`coupled`) vs. Pipeline 4 (`online_coupled`)

Para una corrida estándar de $T_{\text{total}} = 12,000$ pasos VMC con `sweeps_per_step = 5` y $B = 512$ caminantes:

| Métrica de Costo / Datos | Pipeline 3 (`coupled` por bloques: 10 ciclos $\times$ 1200 VMC + 400 RL + 400 Polish) | Pipeline 4 (`online_coupled` paso a paso en 4 Regímenes) | Ventaja de Pipeline 4 |
| :--- | :---: | :---: | :---: |
| **Barridos MCMC usados para VMC** | $12,000 \times 5 = 60,000$ | $12,000 \times 5 = 60,000$ | Mismo presupuesto variacional |
| **Barridos MCMC extra solo para RL (con $\Psi_\theta$ congelada)** | $10 \times 400 + 400 = 4,400$ | **$0$** | **Ahorro del 100% en sweeps MCMC extra** |
| **Pasos con recolección de transiciones RL** | $4,400$ pasos (aislados en bloques) | $10,000$ pasos continuos (Regímenes 2 y 3) | **$2.27\times$ más pasos temporales muestreados** |
| **Desfase entre $\Psi_\theta$ y el Actor $\pi_\phi$** | Hasta $1,200$ pasos VMC sin adaptar $\pi_\phi$ | **$1$ solo paso VMC** (adaptación continua) | **Elimina el desfase entre ciclos** |
| **Estado del Crítico al activar el Actor** | Aleatorio en el Ciclo 1 | **Pre-entrenado gratis sobre $a=\mathbf{0}$ en Régimen 2** | **Sin colapso inicial de política** |

---

## 6. Guía Completa de Ejecución en Ayler (3 GPUs) y Baker (1 GPU)

### 6.1. Preparación del Entorno y Git en Ayler

```bash
ssh ayler
cd ~/RL-VMC-main   # o la ruta de tu repositorio en Ayler
git pull origin main
conda activate jax-gpu
```

### 6.2. Pipeline A: Búsqueda de Hiperparámetros y Ablación de los 4 Regímenes (`run_pipeline4_param_search.sh`)

Este script está diseñado para probar primero un sistema en concreto o un grupo pequeño de 2–3 sistemas representativos, evaluando las baselines (`VMC_MALA`, `P3_BlockCoupled`), la ablación de los 4 regímenes (`RegimeA_ColdStart`, `RegimeB_BlindWarmup`, `RegimeC_NoTailFreeze`, `RegimeD_Full4Regimes`) y la grilla de hiperparámetros del Régimen D:

1. **Correr en Ayler (3 GPUs, 2 trabajos por GPU = 6 trabajos simultáneos) sobre 2 sistemas representativos (`N=4, g=4.0` y `N=6, g=1.0`, default):**
   ```bash
   chmod +x run_pipeline4_param_search.sh
   nohup ./run_pipeline4_param_search.sh \
       --systems "N4:4.0 N6:1.0" \
       --preset fast \
       --gpus "0 1 2" \
       --jobs-per-gpu 2 \
       > master_p4_search_ayler.log 2>&1 &
   ```
2. **Correr la búsqueda sobre UN SOLO sistema en concreto (ej. Cristal de Wigner `N=4, g=4.0`) o 3 sistemas (`N4:2.0 N4:4.0 N6:1.0`):**
   ```bash
   # Un solo sistema en concreto:
   nohup ./run_pipeline4_param_search.sh \
       --systems "N4:4.0" \
       --preset fast \
       --gpus "0 1 2" \
       --jobs-per-gpu 2 \
       results_p4_search_N4_g4.0 > master_p4_search_N4.log 2>&1 &

   # Tres sistemas piloto:
   nohup ./run_pipeline4_param_search.sh \
       --systems "N4:2.0 N4:4.0 N6:1.0" \
       --preset fast \
       --gpus "0 1 2" \
       --jobs-per-gpu 2 \
       results_p4_search_3systems > master_p4_search_3sys.log 2>&1 &
   ```
3. **Correr únicamente la ablación de los 4 regímenes (Baselines + Casos A, B, C y D, muy rápido):**
   ```bash
   nohup ./run_pipeline4_param_search.sh \
       --systems "N4:4.0 N6:1.0" \
       --only-ablations \
       --gpus "0 1 2" \
       --jobs-per-gpu 2 \
       results_p4_ablations > master_p4_ablations.log 2>&1 &
   ```
4. **Personalizar las grillas de hiperparámetros desde la terminal:**
   ```bash
   ./run_pipeline4_param_search.sh \
       --systems "N4:4.0" \
       --schedule-grid "300:300:900 500:500:1200 800:400:1200" \
       --replay-grid "15000 25000 50000" \
       --walkers-store-grid "32 64 128" \
       --utd-grid "1 2" \
       --amp-grid "0.10 0.15" \
       --pen-grid "0.15 0.25" \
       --gpus "0 1 2" --jobs-per-gpu 2
   ```

### 6.3. Pipeline B: Benchmark Multi-Sistema $(N, g)$ (`run_pipeline4_benchmark.sh` y `run_production_benchmark_3pipelines.sh`)

1. **Correr Pipeline 4 sobre el grupo piloto de 3 sistemas (`N4:2.0`, `N4:4.0`, `N6:1.0`, default de `run_pipeline4_benchmark.sh`) en Ayler:**
   ```bash
   chmod +x run_pipeline4_benchmark.sh
   nohup ./run_pipeline4_benchmark.sh \
       --gpus "0 1 2" \
       --jobs-per-gpu 2 \
       > master_p4_pilot3_ayler.log 2>&1 &
   ```
2. **Comparar directamente Pipeline 3 (`coupled`) vs. Pipeline 4 (`online_coupled`) en los mismos sistemas:**
   ```bash
   nohup ./run_pipeline4_benchmark.sh \
       --with-p3 \
       --systems "N4:2.0 N4:4.0 N6:1.0" \
       --gpus "0 1 2" \
       --jobs-per-gpu 2 \
       results_p3_vs_p4_benchmark > master_p3_vs_p4.log 2>&1 &
   ```
3. **Correr Pipeline 4 sobre toda la grilla de producción ($N \in \{2, 4, 6, 10\}$, $g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0\}$):**
   ```bash
   nohup ./run_pipeline4_benchmark.sh \
       --full-grid \
       --gpus "0 1 2" \
       --jobs-per-gpu 2 \
       results_p4_full_production > master_p4_full_ayler.log 2>&1 &
   ```
4. **Regenerar únicamente figuras y CSVs (`--postprocess-only`) o compilar en PDF a 600 DPI:**
   ```bash
   # En PNG (por defecto @ 300 DPI):
   ./run_pipeline4_param_search.sh results_pipeline4_param_search --postprocess-only

   # En PDF de calidad de artículo (@ 600 DPI):
   ./run_pipeline4_param_search.sh results_pipeline4_param_search --postprocess-only --pdf --dpi 600
   ./run_pipeline4_benchmark.sh results_pipeline4_benchmark --postprocess-only --pdf --dpi 600
   ```

### 6.4. ¿Dónde quedan los CSVs ligeros para compartir resultados sin enviar PDFs?

Tanto `run_pipeline4_param_search.sh` como `run_pipeline4_benchmark.sh` concentran **todos los archivos CSV** en una única carpeta ligera `<CARPETA_SALIDA>/results_csv/`:
- `pipeline4_param_search_ranked_summary.csv`: Ranking completo ordenado por energía variacional y $D_{\text{KL}}^{\text{spin}}$ con tiempos de cómputo crudos.
- `live_p4_search_comparison.csv` / `live_pipeline4_online_comparison.csv` / `live_all_models_master.csv`: Tablas maestras comparativas con barras de error Jackknife/multi-cadena.
- `summary_metrics_*.csv` y `production_metrics_*.csv`: Métricas por sistema y por corrida individual.
- `training_history_*.csv`: Evolución paso a paso de `Step`, `Phase` ($1, 2, 3, 4$), `Elapsed_Sec`, `Energy_Mean`, `Energy_Var`, `Acceptance`, `ESJD`, `RL_Reward`, `RL_Reward_EMA`, `Actor_Loss`, `Critic_Loss`, `Q_Val`.
- `observables_*.csv` y `casscf_exact_density_*.csv`: Perfiles espaciales $\rho_{\text{tot}}(x), \rho_\uparrow(x), \rho_\downarrow(x)$ y funciones de correlación de pares $g(r)$.
- `figure_data_acf_energy_*.csv`, `figure_data_kl_convergence_*.csv`, `figure_data_walker_traces_*.csv`, `figure_data_all_models_summary.csv`: Datos crudos exactos de las curvas graficadas en cada figura.
