# Reporte Técnico y Científico: Benchmark de Estrategias de Inicialización (MALA Clásico vs. RL-MALA)

**Framework:** `rl_vmc` — Deep Deterministic Quantum Monte Carlo (1D Interacting Quantum Dots & Wigner Molecules)  
**Fecha del Análisis:** 18 de noviembre de 2026 (Campaña v1 ejecutada: 5–11 de noviembre de 2026 en GPU 2)  
**Orquestador y Graficador Actualizados:** [`run_benchmark_initializations.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_benchmark_initializations.sh), [`scripts/plot_initialization_benchmark.py`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/scripts/plot_initialization_benchmark.py)  
**Referencias Modernas Alineadas:** [`run_advisor_casscf_pipeline.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_advisor_casscf_pipeline.sh), [`NODAL_CROSSING_BENCHMARKS_REPORT.md`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/NODAL_CROSSING_BENCHMARKS_REPORT.md), [`PIPELINE4_ONLINE_COUPLED_REPORT.md`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/PIPELINE4_ONLINE_COUPLED_REPORT.md)

---

## 1. Resumen Ejecutivo

Este documento analiza de forma integral los resultados experimentales de la primera campaña del **Benchmark de Estrategias de Inicialización** alojada en `results_init_benchmark/` (60 corridas independientes: 30 para $N=4$ y 30 para $N=8$; 5 semillas $\times$ 5 cadenas MCMC de $15{,}000$ pasos y $128$ caminantes = $300$ cadenas evaluadas en total), realiza una auditoría forense de los problemas numéricos y de referencia que afectaron a las simulaciones de $N=8$ y a las inicializaciones no estructuradas de $N=4$, y especifica el protocolo modernizado para **re-ejecutar el benchmark sobre funciones de onda CASSCF rigurosamente convergidas (`CASSCF_csv/`)**.

### Hallazgos Clave
1. **Validación limpia en $N=4, g=1.0$ (CASSCF convergido, $E_{\mathrm{ref}} = 1.94862194\text{ Ha}$):**
   - Bajo **`alternating_spin`** (*1 electrón por Gaussiana*), **25/25 cadenas convergen en ambos modelos**, pero **MALA clásico queda topológicamente polarizado** por espín ($D_{\mathrm{KL}}^{\mathrm{spin}} = 0.1006 \pm 0.0013$): la densidad total coincide con CASSCF, pero $\rho_\uparrow(x)$ y $\rho_\downarrow(x)$ permanecen segregados en sus pozos iniciales. **RL-MALA** rompe las barreras de permutación, reduciendo el error de simetría de espín en **$144\times$** ($D_{\mathrm{KL}}^{\mathrm{spin}} = 0.0007 \pm 0.0002$), triplicando el salto cuadrático medio ($\mathrm{ESJD} = 2.038$ vs $0.691\text{ Bohr}^2$) y duplicando la eficiencia real de muestras por segundo ($\mathrm{ESS/s}$ real: $2{,}520$ vs $1{,}225\text{ s}^{-1}$, factor $2.06\times$).
   - Bajo **`symmetric_gaussian`** ($\mathcal{N}(0, 6^2)$) y **`uniform`** ($\mathcal{U}(-10, 10)$), **MALA sufre colapso severo de convergencia**: únicamente **20% (5/25)** de sus cadenas convergen dentro de $2\text{ mHa}$ en `symmetric_gaussian` ($\Delta E = +183.02 \pm 20.18\text{ mHa}$, $\tau_{\mathrm{int}}(E)=474.1$) y apenas **12% (3/25)** en `uniform` ($\Delta E = +173.78 \pm 20.96\text{ mHa}$, $\tau_{\mathrm{int}}(E)=554.2$). En contraste, **RL-MALA** duplica y cuadruplica la tasa de recuperación de cadenas convergidas (**44% = 11/25** en gaussiana y **48% = 12/25** en uniforme), reduciendo el sesgo energético promedio en $\sim 2.5\times$ ($\Delta E = +75.03$ y $+70.40\text{ mHa}$) y la varianza energética en $\sim 3\times\text{–}7\times$.
2. **Diagnóstico forense de $N=8$ en `results_init_benchmark/` (Campaña v1 invalidada para física de $g=1.0$):**
   - **Referencia CASSCF errónea y no convergida:** El archivo utilizado (`figures/casscf_state_cas_n8_g1_omega0.1_soft0.1_b256_c0-a8_det70.npz`), pese a su nombre de archivo, correspondía en realidad a un cálculo con **`g = 4.0`**, **`converged = False`** (detenido al límite de $2{,}000$ iteraciones L-BFGS con gradiente orbital $|g_{\mathrm{orb}}|_{\max} = 3.35 \times 10^{-4} > 10^{-5}$), **core congelado `c2-a10`** ($n_{\mathrm{core}}=2$, $n_{\mathrm{act}}=10$, $2{,}025$ determinantes) y $E = 21.72305163\text{ Ha}$. Al cargarlo, `cli/run_1d.py` sobrescribió automáticamente $g=1.0 \to 4.0$.
   - **Combinación con tres artefactos históricos previos al `HEAD` actual:** (i) el piso aditivo `+ 1e-15` en `CASSCF1DAnsatz` (previo al commit `3a98960`), (ii) la ausencia de calibración dimensional de la varianza estocástica $\sigma_{\mathrm{scale}} = 1.10 \to 0.90$ para 8D (commit `d58bc70`), junto con tasas de aprendizaje bajas con decaimiento coseno (`1e-5/3e-5`), y (iii) el bug histórico de `wall_time_sec = 0.2s` en `diagnostics.py` que infló los valores absolutos de `ESS/s`.
3. **Protocolo Modernizado para la Campaña v2 con Estados Convergidos (`CASSCF_csv/`):**
   - Se actualizaron [`run_benchmark_initializations.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_benchmark_initializations.sh) y [`scripts/plot_initialization_benchmark.py`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/scripts/plot_initialization_benchmark.py) incorporando verificación criptográfica/estructural pre-vuelo de `metadata_json.npy`, los estados convergidos de espacio activo completo (`c0-a4`, `c0-a6`, `c0-a8`, `converged=True`, $b=256$, $g=1.0$), calibración dimensional ($\sigma_{\mathrm{scale}} = 0.90$ en $N=8$), tasas constantes TD3 (`1e-4 / 3e-4`, `--no-use_lr_scheduler`), factorización exacta de determinantes únicos (`b269256`), e inclusión de la inicialización moderna **`randomized_wigner`**.

---

## 2. Marco Físico, Estrategias de Inicialización y Arquitectura de Muestreo

### 2.1 Hamiltoniano Físico y Simetría de Espín
Evaluamos un punto cuántico armónico unidimensional de $N = N_\uparrow + N_\downarrow$ electrones en régimen cerrado ($N_\uparrow = N_\downarrow = N/2$, frecuencia $\omega = 0.1$, suavizado *soft-Coulomb* $a = 0.1$):

$$\hat{H} = \sum_{i=1}^{N} \left( -\frac{1}{2}\frac{\partial^2}{\partial x_i^2} + \frac{1}{2}\omega^2 x_i^2 \right) + \sum_{1 \le i < j \le N} \frac{g}{\sqrt{(x_i - x_j)^2 + a^2}}$$

La función de onda multiconfiguracional congelada $\Psi_{\mathrm{CASSCF}}(\mathbf{x})$ satisface invariancia exacta ante intercambio de espín:

$$\rho_\uparrow^{\mathrm{exact}}(x) = \rho_\downarrow^{\mathrm{exact}}(x) = \frac{1}{2}\rho_{\mathrm{tot}}^{\mathrm{exact}}(x), \qquad D_{\mathrm{KL}}^{\mathrm{spin}} = \frac{1}{2}\Big[D_{\mathrm{KL}}\!\big(p_\uparrow^{\mathrm{MCMC}} \parallel p_\uparrow^{\mathrm{CASSCF}}\big) + D_{\mathrm{KL}}\!\big(p_\downarrow^{\mathrm{MCMC}} \parallel p_\downarrow^{\mathrm{CASSCF}}\big)\Big]$$

Un muestreador verdaderamente ergódico debe converger hacia la energía exacta $\langle E_L \rangle \to E_{\mathrm{CASSCF}}$ **y** recuperar $D_{\mathrm{KL}}^{\mathrm{spin}} \to 0$ independientemente de cómo se inicialicen los caminantes $\mathbf{x}^{(0)}$ al inicio del entrenamiento y de la evaluación.

### 2.2 Estrategias de Inicialización Evaluadas (`MHSampler1D.init_state`)
En ambos modos (`--eval_init_mode same_as_sampler`), tanto el entrenamiento del agente RL como las cadenas finales de evaluación arrancan desde la misma familia inicial:

1. **`alternating_spin` (*1 electrón por Gaussiana*):** Coloca $N$ centros espaciados simétricamente con espines intercalados ($\uparrow, \downarrow, \uparrow, \downarrow, \dots$) escalados al radio de la trampa armónica ($\text{spread} \propto \omega^{-1/2}(1+0.25g)^{1/4}\sqrt{N/2.5}$), más fluctuación gaussiana local $\sigma_{\mathrm{jitter}} = 0.35\,\ell_0/\sqrt{N}$. Representa la inicialización física guiada por red cristalina.
2. **`symmetric_gaussian` (*Gaussiana isotrópica centrada en $x=0$*):** Muestrea $\mathbf{x}^{(0)} \sim \mathcal{N}(\mathbf{0}, \sigma_{\mathrm{env}}^2 \mathbf{I}_N)$ sin conocimiento alguno de la estructura de picos ni ordenamiento de espín ($\sigma_{\mathrm{env}} = 6.0\text{ Bohr}$ para $N=4$; $9.0\text{ Bohr}$ para $N=8$).
3. **`uniform` (*Distribución uniforme en caja confinante*):** Muestrea $\mathbf{x}^{(0)} \sim \mathcal{U}(-L, L)^N$ sin sesgo armónico hacia el centro ($L = 10.0\text{ Bohr}$ para $N=4$; $15.0\text{ Bohr}$ en v1 y $16.0\text{ Bohr}$ en v2 para $N=8$).
4. **`randomized_wigner` (*Estándar moderno v2, commit `d58bc70`*):** Genera sitios de Wigner centrados en $x=0$ con espaciado físico $d_{\mathrm{adj}}(\omega, g, N)$, aplica una **permutación aleatoria independiente por cada caminante** (`argsort(U(0,1))`) y agrega jitter local ($0.12\,d_{\mathrm{adj}}$). Combina equilibrio energético cercano al cristal de Wigner con simetría estadística exacta de espín entre caminantes.

### 2.3 Muestreador Neuronal RL-MALA vs. MALA Clásico
- **MALA Clásico:** Propuesta difusiva local $\mathbf{x}' = \mathbf{x} + \tau \nabla_{\mathbf{x}}\ln|\Psi(\mathbf{x})| + \sqrt{2\tau}\,\boldsymbol{\xi}$ evaluada en modo `eval_only`.
- **RL-MALA (TD3 DeepSet):** Red equivariante por permutaciones [`DeepSetNeuralProposal1D`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/rl_vmc/proposals/proposals_1d.py) con canales relacionales diferenciados por espín ($\uparrow\uparrow/\downarrow\downarrow$ y $\uparrow\downarrow$), parametrizando una corrección residual en deriva y escala:
  $$\boldsymbol{\mu}_\theta(\mathbf{x}) = \mathbf{x} + \tau\,\mathbf{g}_{\mathrm{clip}}(\mathbf{x}) + A_\mu \tanh(\mathbf{u}_\theta), \qquad \boldsymbol{\sigma}_\theta(\mathbf{x}) = \sqrt{2\tau}\,\exp\!\big(\operatorname{clip}(s_\sigma \tanh(s_\theta), -5, 1)\big)$$
  entrenada fuera de política con TD3 maximizando la recompensa **CDLB** (*Contrastive Divergence Lower Bound*) con penalización de rechazo $\lambda_{\mathrm{pen}} = 0.25$ y filtro estricto de Metropolis-Hastings.

---

## 3. Auditoría de Referencias e Infraestructura de la Campaña v1 (`results_init_benchmark/`)

La campaña ejecutada entre el 5 y el 11 de noviembre de 2026 (`results_init_benchmark/logs/master_benchmark.log`) corrió secuencialmente en GPU 2 bajo el protocolo `N_exp = 5` semillas (`42, 1042, 2042, 3042, 4042`), `N_eval = 5` cadenas por réplica (`burn_in = 1,000`, `measure_steps = 15,000`, `batch_size = 128`) y `train_steps = 4,000` para RL-MALA:

| Sistema Nominal | Archivo `.npz` Utilizado en Campaña v1 | Estado Real en Metadata `.json` | Parámetros Físicos Reales | $E_{\mathrm{CASSCF}}$ Referencia (Ha) | Validez Científica |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`N4_w0.1_g1.0`** | `figures/casscf-state_n4_g1_omega0.1_soft0.1_b256_c0-a4_det6.npz` | `converged=True` ($58$ iters L-BFGS, $\|g_{\mathrm{orb}}\|=8.74\times 10^{-6}$) | $N=4$, **$g=1.0$**, $b=256$, $c0\text{-}a4$, $6/36$ dets | $1.94862194$ | **Válida** (CASSCF convergido exacto; afectada solo por piso `+1e-15` en `sym_gauss`/`uniform`) |
| **`N8_w0.1_g1.0`** *(corrió como `N8_g4.0`)* | `figures/casscf_state_cas_n8_g1_omega0.1_soft0.1_b256_c0-a8_det70.npz` | **`converged=False`** ($2{,}000$ iters max., $\|g_{\mathrm{orb}}\|=3.35\times 10^{-4}$) | $N=8$, **$g=4.0$**, $b=256$, **`c2-a10`**, $2{,}025$ dets | $21.72305163$ | **Inválida para $g=1.0$** (estado truncado $c2\text{-}a10$ no convergido a $g=4.0$) |

---

## 4. Análisis Cuantitativo Completo: Sistema Convergido $N=4, g=1.0$

### 4.1 Tabla Estadística Agregada (5 Réplicas $\times$ 5 Cadenas = 25 Cadenas por Configuración)

La siguiente tabla combina las métricas de `results_init_benchmark/figures/initialization_benchmark_summary_table.csv` con el conteo por cadena extraído de los 30 archivos de log de $N=4$ y los tiempos reales de reloj en GPU (`~140–157 s/cadena`, corrigiendo el defecto histórico de `0.2 s` en `diagnostics.py`):

| Modelo | Inicialización | Cadenas Convergidas ($|\Delta E| < 2\text{ mHa}$) | $\Delta E$ vs CASSCF (mHa) | $\mathrm{Var}(E)$ ($\mathrm{Ha}^2$) | $\tau_{\mathrm{int}}(E)$ (pasos) | Aceptación (%) | $\mathrm{ESJD}$ ($\mathrm{Bohr}^2$) | $\mathrm{ESS/s}$ Real* ($\mathrm{s}^{-1}$) | $D_{\mathrm{KL}}^{\mathrm{tot}}$ | $D_{\mathrm{KL}}^{\mathrm{spin}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MALA** | `alternating_spin` | **25 / 25 (100%)** | $+0.32 \pm 0.09$ | $0.035 \pm 0.000$ | $10.02 \pm 0.20$ | $88.7\% \pm 0.0\%$ | $0.691 \pm 0.000$ | $1{,}225.3 \pm 23.5$ | $3.58\times 10^{-5}$ | $0.1006 \pm 0.0013$ |
| **RL-MALA** | `alternating_spin` | **25 / 25 (100%)** | **$+0.11 \pm 0.02$** | **$0.034 \pm 0.000$** | **$5.04 \pm 0.07$** | $52.9\% \pm 1.5\%$ | **$2.038 \pm 0.052$** | **$2{,}520.3 \pm 28.5$** | $3.34\times 10^{-5}$ | **$\mathbf{0.0007 \pm 0.0002}$** |
| **MALA** | `symmetric_gaussian` | 5 / 25 (20%) | $+183.02 \pm 20.18$ | $6.643 \pm 4.173$ | $474.10 \pm 77.94$ | $87.4\% \pm 0.2\%$ | $0.681 \pm 0.001$ | $1{,}151.4 \pm 23.7$ | $2.38\times 10^{-3}$ | $0.0058 \pm 0.0004$ |
| **RL-MALA** | `symmetric_gaussian` | **11 / 25 (44%)** | **$+75.03 \pm 23.99$** | **$0.932 \pm 0.349$** | **$214.65 \pm 193.15$** | $52.5\% \pm 1.7\%$ | **$2.010 \pm 0.063$** | **$2{,}343.8 \pm 217.0$** | $1.21\times 10^{-3}$ | **$0.0023 \pm 0.0006$** |
| **MALA** | `uniform` | 3 / 25 (12%) | $+173.78 \pm 20.96$ | $2.092 \pm 0.392$ | $554.19 \pm 84.51$ | $87.3\% \pm 0.1\%$ | $0.681 \pm 0.001$ | $976.2 \pm 74.8$ | $2.21\times 10^{-3}$ | $0.0056 \pm 0.0004$ |
| **RL-MALA** | `uniform` | **12 / 25 (48%)** | **$+70.40 \pm 24.91$** | **$0.752 \pm 0.253$** | **$241.47 \pm 139.52$** | $52.3\% \pm 1.6\%$ | **$2.017 \pm 0.059$** | **$2{,}274.0 \pm 166.4$** | $1.08\times 10^{-3}$ | **$0.0020 \pm 0.0006$** |

*\*Nota metodológica sobre $\mathrm{ESS/s}$:* En la versión del código usada entre el 5 y 11 de noviembre, `diagnostics.py` asignaba un `wall_time_sec = 0.2 s` ficticio por cadena, escalando la columna cruda de la tabla CSV por $\sim 700\times$ (reportando $962{,}891$ vs $1{,}919{,}577\text{ s}^{-1}$ en `alternating_spin`). Aquí reportamos $\mathrm{ESS/s}_{\mathrm{real}} = \mathrm{ESS} / t_{\mathrm{cadena}}$ usando los tiempos reales de GPU registrados en log ($139.3\text{–}157.2\text{ s}$); la razón relativa $G_{\mathrm{RL}} \approx 2.06\times$ (`alt_spin`) a $2.33\times$ (`uniform`) se conserva invariante y el bug ya fue corregido en `HEAD`.

### 4.2 Interpretación Física y Análisis de Trazas y Densidades ($N=4, g=1.0$)

![Panel Comparativo de Barras N4_g1.0](results_init_benchmark/figures/benchmark_initialization_N4_g1.0.png)  
*Figura 4.1: Métricas estadísticas agregadas (Media $\pm$ SEM sobre 5 réplicas) para $N=4, g=1.0$ comparando MALA Clásico (naranja) frente a RL-MALA TD3 (verde) bajo las tres estrategias de inicialización.*

1. **Ruptura del bloqueo topológico de espín en `alternating_spin` ($144\times$ mejora en $D_{\mathrm{KL}}^{\mathrm{spin}}$):**
   - Aunque MALA reproduce perfectamente la densidad total ($\rho_{\mathrm{tot}}(x)$, línea negra discontinua) y obtiene $\Delta E = +0.32\text{ mHa}$, sus histogramas resueltos por espín exhiben **segregación severa izquierda-derecha** (Figura 4.2, arriba izquierda: el pozo en $x \approx -7.7\text{ Bohr}$ es $\sim 85\%$ espín $\uparrow$ azul y el pozo en $x \approx +7.7\text{ Bohr}$ es $\sim 85\%$ espín $\downarrow$ rojo), arrojando $D_{\mathrm{KL}}^{\mathrm{spin}} = 0.1006$. En la traza de un caminante individual (Figura 4.3, arriba izquierda), MALA ejecuta apenas un intercambio cerca del paso $10{,}000$.
   - Por el contrario, RL-MALA ejecuta decenas de intercambios colectivos de múltiples partículas simultáneamente (Figura 4.3, arriba derecha), restaurando la simetría exacta $\rho_\uparrow(x) = \rho_\downarrow(x)$ con $D_{\mathrm{KL}}^{\mathrm{spin}} = 0.0007$ (**$144\times$ menor**) y duplicando la eficiencia de muestreo ($\tau_{\mathrm{int}}(E) = 5.04$ vs $10.02$).

![Distribuciones de Densidad Muestreadas vs CASSCF Exacto N4_g1.0](results_init_benchmark/figures/benchmark_densities_vs_casscf_N4_g1.0.png)  
*Figura 4.2: Densidades de 1 cuerpo muestreadas ($\rho_\uparrow$, $\rho_\downarrow$, $\rho_{\mathrm{tot}}$) comparadas con la referencia analítica CASSCF exacta (verde) para el Mejor MALA (izquierda) y Mejor RL-MALA (derecha) de cada inicialización en $N=4, g=1.0$.*

![Trazas Individuales de 1 Caminante N4_g1.0](results_init_benchmark/figures/benchmark_walker_traces_N4_g1.0.png)  
*Figura 4.3: Trazas temporales de un único caminante físico (Cadena 0, Caminante 0) y distribución marginal con referencia CASSCF exacta en $N=4, g=1.0$. Nótese cómo MALA permanece congelado en los pozos de espín mientras RL-MALA permuta activamente en los tres regímenes.*

2. **Resiliencia en `symmetric_gaussian` y `uniform`, y el origen de las bandas energéticas elevadas ($\approx 2.03, 2.10, 2.23, 2.44\text{ Ha}$):**
   - Al inspeccionar los logs cadena por cadena de los mejores y peores modelos en `symmetric_gaussian` y `uniform`, se observa un patrón discreto notable: las cadenas no convergen de forma continua alrededor de $2.13\text{ Ha}$, sino que o bien alcanzan con precisión el estado base (**$1.9486\text{–}1.9498\text{ Ha}$**), o quedan desplazadas en escalones discretos cercanos a **$2.03$, $2.10$, $2.23$ o $2.44\text{ Ha}$** (por ejemplo, en `RL_MALA_uniform_exp3`, 4 cadenas logran $1.9481\text{–}1.9490\text{ Ha}$ mientras la Cadena 0 marca $2.1009\text{ Ha}$; en `RL_MALA_uniform_exp5`, todas las cadenas marcan $2.0976\text{–}2.1122\text{ Ha}$), disparando $\mathrm{Var}(E)$ de $0.034\text{ Ha}^2$ a $0.75\text{–}6.64\text{ Ha}^2$.
   - **Mecanismo físico/numérico:** Como ambos métodos comparten exactamente las mismas semillas iniciales de evaluación (`seed = 42 + 1000*(exp-1)`), cuando la inicialización aleatoria `N(0, 6^2)` o `U(-10, 10)` coloca uno o pocos caminantes (de los 128 del lote) en regiones alejadas o traslapadas donde $|\Psi_{\mathrm{CASSCF}}(\mathbf{x})| < 10^{-15}$, la implementación previa al commit `3a98960` (piso aditivo `+ 1e-15`) anulaba la deriva armónica ($\mathbf{F}_Q = \mathbf{0}$), atrapando a esos caminantes externos con $V_{\mathrm{HO}} \sim 10\text{–}20\text{ Ha}$ (desplazando el promedio de los 128 caminantes en múltiplos de $\sim +0.08\text{ a }+0.15\text{ Ha}$ por caminante fugado!).
   - A pesar de esa adversidad compartida desde el paso $t=0$, **RL-MALA recupera hasta $4\times$ más cadenas libres de atrapamiento en inicialización uniforme ($48\%$ vs $12\%$) y $2.2\times$ en gaussiana ($44\%$ vs $20\%$)** gracias a su movilidad triplicada ($\mathrm{ESJD} \approx 2.02$ vs $0.68\text{ Bohr}^2$) durante el burn-in de $1{,}000$ pasos.
   - Asimismo, el menor valor aparente de $D_{\mathrm{KL}}^{\mathrm{spin}} \approx 0.0056$ de MALA en estas inicializaciones proviene del promedio simétrico entre caminantes inicializados en distintos sectores del lote, **no** de cruces dinámicos (como confirma el panel inferior izquierdo de la Figura 4.3, donde el caminante de MALA uniforme exhibe **cero** permutaciones de espín en $15{,}000$ pasos).

---

## 5. Diagnóstico Forense de $N=8$ (`results_init_benchmark/N8_w0.1_g1.0/`)

### 5.1 Resultados Observados en la Campaña v1 ($N=8$, corrió realmente a $g=4.0$)

| Modelo | Inicialización | $\Delta E$ vs Ref. (`21.723 Ha`) (mHa) | $\mathrm{Var}(E)$ ($\mathrm{Ha}^2$) | $\tau_{\mathrm{int}}(E)$ | Aceptación (%) | $\mathrm{ESJD}$ ($\mathrm{Bohr}^2$) | $\mathrm{ESS/s}$ Real ($t \approx 2697\text{s}$) | Mejor Réplica ($\Delta E$, Acc, ESJD) | Peor Réplica ($\Delta E$, Acc, ESJD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **MALA** | `alt_spin` | $+565.2 \pm 207.4$ | $178.3 \pm 68.6$ | $1097.0 \pm 35.7$ | $82.7\%$ | $1.040$ | $0.67 \pm 0.02$ | `exp4`: $+144\text{ mHa}$, $82.5\%$, $1.037$ | `exp2`: $+1299\text{ mHa}$, $82.7\%$, $1.039$ |
| **RL-MALA** | `alt_spin` | $+2958.8 \pm 1419.9$ | $185.9 \pm 76.1$ | $533.0 \pm 397.0$ | $32.0\%$ | $1.370 \pm 0.542$ | $17.7 \pm 8.2$ | `exp5`: $-142\text{ mHa}$, $33.8\%$, **$2.631$** | `exp1`: $+7006\text{ mHa}$, $30.6\%$, $0.130$ |
| **MALA** | `sym_gauss` | $+941.3 \pm 464.1$ | $506.1 \pm 202.5$ | $779.7 \pm 44.9$ | $80.9\%$ | $1.016$ | $1.20 \pm 0.09$ | `exp3`: $+55\text{ mHa}$, $81.5\%$, $1.025$ | `exp4`: $+2207\text{ mHa}$, $80.1\%$, $1.007$ |
| **RL-MALA** | `sym_gauss` | $+2716.6 \pm 1154.2$ | $513.9 \pm 212.1$ | $1049.9 \pm 432.4$ | $27.6\%$ | $1.258 \pm 0.566$ | $13.6 \pm 8.1$ | `exp3`: $-488\text{ mHa}$, $32.7\%$, **$2.570$** | `exp2`: $+5812\text{ mHa}$, $18.1\%$, $0.067$ |
| **MALA** | `uniform` | $+2143.9 \pm 779.8$ | $1960.4 \pm 836.1$ | $786.6 \pm 102.3$ | $80.6\%$ | $1.013$ | $1.20 \pm 0.17$ | `exp4`: $-66\text{ mHa}$, $80.1\%$, $1.008$ | `exp5`: $+4401\text{ mHa}$, $79.4\%$, $0.998$ |
| **RL-MALA** | `uniform` | $+3343.2 \pm 1289.7$ | $1905.1 \pm 846.3$ | $1230.4 \pm 485.5$ | $28.7\%$ | $1.292 \pm 0.550$ | $11.5 \pm 6.9$ | `exp3`: $+49\text{ mHa}$, $32.8\%$, **$2.581$** | `exp2`: $+6825\text{ mHa}$, $22.4\%$, $0.201$ |

![Panel Comparativo N8_g4.0](results_init_benchmark/figures/benchmark_initialization_N8_g4.0.png)  
*Figura 5.1: Resultados de la campaña v1 en $N=8$ (que ejecutó realmente sobre el estado no convergido a $g=4.0$). Nótense las barras de error gigantescas en RL-MALA producto de promediar 2 réplicas exitosas (`exp3, exp5`) con 3 réplicas que sufrieron colapso dimensional (`exp1, exp2, exp4`).*

![Densidades vs CASSCF N8_g4.0](results_init_benchmark/figures/benchmark_densities_vs_casscf_N8_g4.0.png)  
*Figura 5.2: Densidades de 1 cuerpo en $N=8$ (`N8_g4.0`). El estado CASSCF truncado `c2-a10` no convergido exhibe únicamente 4 picos centrales y colas planas no físicas más allá de $[-20, +20]\text{ Bohr}$, evidenciando la causa estructural de las varianzas energéticas extremas.*

![Trazas de Caminante N8_g4.0](results_init_benchmark/figures/benchmark_walker_traces_N8_g4.0.png)  
*Figura 5.3: Trazas de 1 caminante en `N8_g4.0`. Obsérvense los 4 electrones externos expulsados hacia las alas no confinadas ($\pm 12\text{ a }\pm 20\text{ Bohr}$) por los orbitales no convergidos del core congelado `c2-a10`.*

![Resumen Aceleración Global](results_init_benchmark/figures/benchmark_initialization_summary_speedups.png)  
*Figura 5.4: Factores de aceleración relativa $G_{\mathrm{RL}}$ y descorrelación $\tau_{\mathrm{int}}(E)_{\mathrm{MALA}} / \tau_{\mathrm{int}}(E)_{\mathrm{RL}}$ de la campaña v1.*

### 5.2 Las Cinco Causas Raíz del Comportamiento Anómalo en $N=8$
La auditoría cruzada de logs, `summary_cas_n8*.json` y gráficas explica completamente por qué $N=8$ mostró varianzas de hasta $1{,}960\text{ Ha}^2$ y comportamiento bimodal en RL-MALA:

1. **Archivo CASSCF erróneo a $g=4.0$, con core congelado `c2-a10` y `converged = False`:**
   - Aunque su nombre decía `n8_g1...c0-a8_det70`, el archivo en `figures/` contenía un cálculo a **$g=4.0$**, detenido sin converger a las $2{,}000$ iteraciones L-BFGS (`|g_orb| = 3.35e-4`), con **2 orbitales de core congelados** (4 electrones atrapados en el centro) y **10 orbitales activos ($2{,}025$ determinantes)**.
   - Como se visualiza en la Figura 5.2, esa expansión mal condicionada generó **4 picos interiores y dos colas extensas no físicas** que desbordan los límites $\pm 20\text{ Bohr}$. Al usar `np.histogram(..., density=True)` dentro de $[-20, 20]$, la densidad muestreada se renormalizó ignorando la cola exterior, provocando que la curva negra muestreada pareciera sistemáticamente más alta que la curva verde de CASSCF (defecto ya corregido en el nuevo `scripts/plot_initialization_benchmark.py`).
   - Peor aún, evaluar energías locales sobre orbitales activos no convergidos produjo fluctuaciones de cadena entre $17.26\text{ Ha}$ (por debajo de la propia energía de referencia $21.72\text{ Ha}$) y $37.19\text{ Ha}$.
2. **Bifurcación bimodal de entrenamiento TD3 (`sigma_action_scale = 1.10` en 8D + `lr = 1e-5`):**
   - Al mantener $\sigma_{\mathrm{scale}} = 1.10$ en $\mathbb{R}^8$ y tasa de aprendizaje pequeña (`1e-5/3e-5` con decaimiento coseno a $5\times 10^{-7}$) en solo $4{,}000$ pasos, las corridas `exp3` (`seed=2042`) y `exp5` (`seed=4042`) sí lograron aprender políticas de salto óptimas (`Reward: -0.59 -> +0.41`, `Acc ≈ 33%`, `ESJD = 2.57–2.63 Bohr^2`), mientras que **`exp1` y `exp2` colapsaron monotónicamente** (`Acc: 74% -> 15%`, `ESJD: 1.0 -> 0.067 Bohr^2`, $\langle E \rangle \approx 27.5\text{–}28.7\text{ Ha}$, es decir $+6\text{ a }+7\text{ Ha}$ de sesgo!). Como diagnosticamos en [`NODAL_CROSSING_BENCHMARKS_REPORT.md`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/NODAL_CROSSING_BENCHMARKS_REPORT.md) (Sección 11.3), en $N=8$ se requiere **$\sigma_{\mathrm{scale}} = 0.90$**.
3. **Ausencia de la factorización de determinantes únicos (`b269256`):**
   - Evaluar $2{,}025$ determinantes por paso sin factorizar en determinantes únicos de espín tomó **$2{,}698\text{ s}$ (~45 minutos) por cadena**, acumulando 6 días continuos de ejecución para el benchmark completo.
4. **Piso aditivo `+ 1e-15` en `ansatz_1d.py`:** Previos a `3a98960`, los caminantes en las colas $|x| > 15\text{ Bohr}$ perdían su gradiente armónico $\mathbf{F}_Q = \mathbf{0}$.
5. **Bug de `wall_time_sec = 0.2s` en `diagnostics.py`:** Infló los números absolutos de $\mathrm{ESS/s}$ reportados el 11 de noviembre (ya corregido en `HEAD`).

---

## 6. Siguiente Benchmark con Estados CASSCF Convergidos y Protocolo Moderno

### 6.1 Estados CASSCF Convergidos Auditados (`CASSCF_csv/`)

En la carpeta `CASSCF_csv/` disponemos de los estados **estrictamente convergidos** en base $N_{\mathrm{basis}}=256$, cuadratura $800$, espacio activo completo ($n_{\mathrm{core}} = 0$) y acoplamiento consistente **$g = 1.0, \omega = 0.1, a = 0.1$**:

| Sistema | Archivo Convergido (`CASSCF_csv/`) | Espacio Activo y Dets | Iter. L-BFGS | Gradiente $\|g_{\mathrm{orb}}\|_{\max}$ | $E_{\mathrm{CASSCF}}$ Exacta (Ha) | Estructura Espacial (Picos / RMS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`N4_w0.1_g1.0`** | `casscf_state_cas_n4_g1.0_omega0.1_soft0.1_b256_c0-a4_det6.npz` | `c0-a4`, $36$ full ($6$ sel.) | $58$ | $8.74\times 10^{-6}$ | **$1.94862194$** | 4 picos ($\pm 2.55, \pm 7.7\text{ Bohr}$) |
| **`N6_w0.1_g1.0`** *(opcional)* | `casscf_state_cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_det-20.npz` | `c0-a6`, $20/400$ dets | $71$ | $7.93\times 10^{-6}$ | **$4.52405213$** | 6 picos ($\pm 1.99, \pm 6.14, \pm 10.86$), $R_{\mathrm{RMS}}=7.36\text{ Bohr}$ |
| **`N8_w0.1_g1.0`** | `casscf_state_cas_n8_g1.0_omega0.1_soft0.1_b256_c0-a8_det-70.npz` | `c0-a8`, $70/4900$ dets | $85$ | $8.70\times 10^{-6}$ | **$8.07323817$** | **8 picos limpios** ($\pm 1.71, \pm 5.19, \pm 8.95, \pm 13.38$), $R_{\mathrm{RMS}}=7.42\text{ Bohr}$ |

### 6.2 Actualización de Protocolo: Cómo lo Hacen los Benchmarks Modernos (`run_advisor_casscf_pipeline.sh` / `HEAD`)

Al integrar todas las mejoras desarrolladas en `run_advisor_casscf_pipeline.sh`, `run_production_benchmark_3pipelines.sh` y los fixes del núcleo `rl_vmc` en `HEAD`, el nuevo benchmark en [`run_benchmark_initializations.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_benchmark_initializations.sh) aplica la siguiente actualización integral:

| Parámetro / Componente | Campaña v1 (`results_init_benchmark`, Nov 5) | Campaña v2 Convergida (`HEAD`) | Propósito Científico / Técnico |
| :--- | :--- | :--- | :--- |
| **Referencias CASSCF** | `figures/...n8_g1...npz` (`g=4.0`, `c2-a10`, `conv=False`) | **`CASSCF_csv/`** (`g=1.0`, `c0-aN`, `conv=True`) + fallback auto-CASSCF CPU | Elimina colas no físicas y valida física real de 8 picos a $g=1.0$. |
| **Verificación Pre-Vuelo** | Sólo existencia del archivo `.npz` | Inspección `metadata_json.npy` + cuarentena automática de `.h5` inválidos | Impide silenciosamente muestrear estados inconclusos o de otro $g$. |
| **Evaluación Ansatz (`HEAD`)** | Piso `+ 1e-15`; sin factorización de determinantes | `tiny_eps + harm_floor_log` (`3a98960`); determinantes únicos (`b269256`) | Erradica fugas en colas (`sym_gauss`/`uniform`) y acelera $N=8$ ($\sim 10\times$). |
| **Estrategias Inicialización** | `alternating_spin`, `symmetric_gaussian`, `uniform` | Las 3 originales **+ `randomized_wigner`** (4 modos configurables) | Contrasta inicializaciones ingenuas vs la cristalina moderna aleatorizada. |
| **Escala Estocástica $\sigma_{\mathrm{scale}}$** | `1.10` en $N=4$ y $N=8$ | `1.10` ($N=4, 6$) y **`0.90` ($N=8$)** | Previene el colapso de aceptación y TD3 en dimensión 8D. |
| **Amplitud Actor $A_\mu$** | `0.10` ($N=4, 8$) | `0.14` ($N=4, 6$ a $g=1.0$) y `0.10` ($N=8$), `--max_phys_jump_factor 2.0` | Maximiza saltos colectivos al espaciado inter-partícula sin exceso de rechazo. |
| **Optimizador TD3** | `lr=1e-5/3e-5` (coseno activo), `4,000` pasos, burn-in `1,000` | **`--no-use_lr_scheduler --lr_actor 1e-4 --lr_critic 3e-4`**, `6,000` pasos, burn-in `2,000` | Aprendizaje rápido y estable sin decaimiento prematuro de tasa. |
| **Evaluación & Almacenaje** | `eval_burn_in=1000`, `measure=15000`, `eloc_chunk=2048` | `eval_burn_in=2000`, `measure=30000`, `eloc_chunk=64`, `step_mean/float32` | Mayor termalización para `uniform`/`sym_gauss`, cero OOM ni ENOSPC. |
| **Estabilidad GPU / XLA** | Sin protección Triton GEMM | `XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_enable_triton_gemm=false"` + *Circuit Breaker* | Evita fallos `ptxas` en Ayler/Baker y aborta ante caídas de entorno. |

### 6.3 Comandos Exactos para Ejecutar el Nuevo Benchmark Convergido

En **Baker** o **Ayler** dentro del entorno conda (`jax-gpu`):

```bash
# 1. Actualizar repositorio y verificar scripts
git pull origin main
conda activate jax-gpu
chmod +x run_benchmark_initializations.sh

# 2. Correr benchmark completo con estados CASSCF convergidos (N4 y N8 a g=1.0)
#    Incluye las 4 inicializaciones (alternating_spin, randomized_wigner, symmetric_gaussian, uniform),
#    5 réplicas x 5 cadenas x 30,000 pasos por sistema.
#    En Ayler (3 GPUs, ej. --gpus "0 1 2" -j 1) o Baker (ej. --gpus "0" o "2"):
nohup ./run_benchmark_initializations.sh \
    --system "N4 N8" \
    --gpus "0 1 2" \
    --jobs-per-gpu 1 \
    --n-exp 5 \
    --n-eval 5 \
    results_init_benchmark_converged > master_init_converged.log 2>&1 &

# Monitoreo en vivo:
tail -f results_init_benchmark_converged/logs/master_benchmark.log
```

**Variantes útiles:**
- **Incluir también el sistema intermedio $N=6, g=1.0$ (`det-20`):** añade `--include-n6` (o `--system "N4 N6 N8"`).
- **Correr únicamente $N=8$ convergido:** añade `--system N8`.
- **Evaluar únicamente las 3 inicializaciones originales** (sin `randomized_wigner`): añade `--classic-3-inits`.
- **Recompilar únicamente figuras en PDF y PNG a 300 DPI sin re-entrenar:**
  ```bash
  ./run_benchmark_initializations.sh results_init_benchmark_converged --postprocess-only --both-formats --dpi 300
  ```
- **Nota sobre `CASSCF_csv/`:** La carpeta `CASSCF_csv/` no se versiona en Git (`.gitignore`), pero `run_benchmark_initializations.sh` busca automáticamente en `CASSCF_csv/`, `results_production_3pipelines*/pipeline1_casscf/` y `figures/`, verificando `converged=True, n_core=0, b>=256`; si no encuentra el archivo localmente en la máquina destino, **calcula automáticamente en CPU** el estado CASSCF convergido exacto mediante `cli/run_casscf_1d.py` (~1.6 s para $N=4$, ~49 s para $N=8$).
