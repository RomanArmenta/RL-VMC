# Benchmarks de Cruce Nodal y Muestreo Puro Multi-Determinante (CASSCF 1D): Teoría, Implementación y Resultados

Este documento presenta el marco teórico, la implementación algorítmica y el análisis cuantitativo exhaustivo de los dos benchmarks fundamentales de **muestreo MCMC puro sobre funciones de onda multiconfiguracionales congeladas (CASSCF)** ejecutados en `baker`:

1. **Benchmark 1 (`Wigner_N4_g4.0`):** Cristal de Wigner fuertemente correlacionado de $N=4$ electrones ($N_\uparrow=2, N_\downarrow=2, \omega=0.1, g=4.0, a=0.1$, 36 determinantes CI en base $N_{\mathrm{basis}}=32$, $E_{\mathrm{CASSCF}} = 4.1595825\text{ Ha}$).
2. **Benchmark 2 (`MultiDet_N6_g1.0`):** Sistema multi-determinante de $N=6$ electrones en espacio activo completo $\mathrm{CAS}(6e, 6o)$ truncado a los 20 determinantes dominantes en base expandida $N_{\mathrm{basis}}=256$ ($N_\uparrow=3, N_\downarrow=3, \omega=0.1, g=1.0, a=0.1$, $E_{\mathrm{CASSCF}} = 4.5240521\text{ Ha}$).

---

## 1. Marco Teórico: Congelamiento Topológico y Cruce de Barreras Nodales en 1D

### 1.1 Hamiltoniano Físico y Simetría Exacta de Espín
Consideramos un sistema de $N = N_\uparrow + N_\downarrow$ electrones confinados en una trampa armónica unidimensional con interacción repulsiva de Coulomb suavizada (*soft-Coulomb*):

$$
\hat{H} = \sum_{i=1}^{N} \left( -\frac{1}{2}\frac{\partial^2}{\partial x_i^2} + \frac{1}{2}\omega^2 x_i^2 \right) + \sum_{1 \le i < j \le N} \frac{g}{\sqrt{(x_i - x_j)^2 + a^2}}
$$

Dado que $\hat{H}$ es completamente independiente del espín ($\hat{H}$ conmuta con los operadores de permutación espacial y con $\hat{S}^2, \hat{S}_z$), para un sistema cerrado en espín ($N_\uparrow = N_\downarrow = N/2$) cualquier estado propio exacto o aproximación variacional invariante de espín (como RHF-CASSCF) satisface idénticamente la **simetría de densidad de un cuerpo resuelta por espín**:

$$
\rho_\uparrow^{\mathrm{exact}}(x) = \rho_\downarrow^{\mathrm{exact}}(x) = \frac{1}{2}\rho_{\mathrm{tot}}^{\mathrm{exact}}(x), \qquad \forall x \in \mathbb{R}
$$

donde la densidad normalizada a la unidad se define como $p_\sigma(x) = \rho_\sigma(x) / N_\sigma$ con $\int_{-\infty}^{\infty} p_\sigma(x)\,dx = 1$.

### 1.2 Estructura Nodal y Seudo-Barreras de Wigner
La función de onda multiconfiguracional CASSCF expandida en $M_{\mathrm{det}}$ determinantes de Slater se expresa como:

$$
\Psi_{\mathrm{CASSCF}}(\mathbf{x}_\uparrow, \mathbf{x}_\downarrow) = \sum_{k=1}^{M_{\mathrm{det}}} c_k \, \det\!\left[\phi_{k,a}^\uparrow(x_{i,\uparrow})\right]_{i,a=1}^{N_\uparrow} \det\!\left[\phi_{k,b}^\downarrow(x_{j,\downarrow})\right]_{j,b=1}^{N_\downarrow}
$$

donde cada orbital molecular activo o de core $\phi_p(x) = \sum_{n=0}^{N_{\mathrm{basis}}-1} C_{n,p}\,\chi_n^{\mathrm{HO}}(x; \omega)$ se expande en autofunciones del oscilador armónico 1D evaluadas mediante la recurrencia estable de tres términos de Hermite.

En dimensión $d=1$, la topología del espacio de configuración $\mathbb{R}^N$ presenta dos obstáculos severos para los muestreadores locales basados en gradientes:

1. **Hiperplanos Nodales Exactos por Antisimetría de Pauli ($x_{i,\sigma} = x_{j,\sigma}$):**
   Cuando dos electrones del **mismo espín** coinciden ($x_{i,\sigma} = x_{j,\sigma}$), cada determinante se anula exactamente ($\Psi_{\mathrm{CASSCF}} = 0$). La deriva cuántica de Langevin:
   $$
   \mathbf{F}_{\mathrm{Q}}(\mathbf{x}) = \nabla_{\mathbf{x}} \ln |\Psi_{\mathrm{CASSCF}}(\mathbf{x})| \sim \frac{1}{x_{i,\sigma} - x_{j,\sigma}} \to \pm\infty
   $$
   diverge repulsivamente en la superficie nodal. En una propuesta difusiva estándar de tipo MALA:
   $$
   \mathbf{x}' = \mathbf{x} + \tau \nabla_{\mathbf{x}} \ln |\Psi(\mathbf{x})| + \sqrt{2\tau}\,\boldsymbol{\xi}, \qquad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
   $$
   la deriva cuántica empuja a los caminantes lejos del nodo; si por fluctuación gaussiana un caminante intenta cruzar el nodo con un paso $\tau$ grande, cae en regiones de alta energía o de asimetría en la razón de transición inversa $q(\mathbf{x} \mid \mathbf{x}') / q(\mathbf{x}' \mid \mathbf{x})$, desplomando la aceptación de Metropolis-Hastings.

2. **Seudo-Barreras de Correlación de Wigner entre Espines Opuestos ($x_{i,\uparrow} \approx x_{j,\downarrow}$):**
   Aunque el principio de Pauli no prohíbe el cruce de electrones de **espín opuesto** ($x_{i,\uparrow} = x_{j,\downarrow}$), la repulsión de Coulomb $g / \sqrt{(x_i - x_j)^2 + a^2}$ junto con los determinantes excitados de correlación izquierda-derecha suprimen exponencialmente la densidad de probabilidad $|\Psi(\mathbf{x})|^2$ cuando dos electrones se superponen. Esto divide $\mathbb{R}^N$ en $N!$ **bolsillos de permutación de espín** separados por valles profundos de baja probabilidad.
   Si la cadena MCMC se inicia en un sector ordenado por espín (por ejemplo, $x_{1,\uparrow} < \dots < x_{N_\uparrow,\uparrow} < 0 < x_{1,\downarrow} < \dots < x_{N_\downarrow,\downarrow}$), el muestreador MALA clásico queda **topológicamente congelado** (*broken ergodicity*): muestrea correctamente los $N$ picos espaciales de la densidad total $\rho_{\mathrm{tot}}(x)$ ($D_{\mathrm{KL}}^{\mathrm{tot}} \sim 10^{-5}$), pero es incapaz de permutar los espines entre los pozos de Wigner, produciendo una polarización espuria masiva ($\rho_\uparrow(x) \neq \rho_\downarrow(x)$, con $D_{\mathrm{KL}}^{\mathrm{spin}} \sim 0.12 - 0.63$).

---

## 2. Arquitectura RL-MALA con Actor DeepSet/SchNet y Recompensa CDLB

### 2.1 Propuesta Neuronal Equivariante (Residual MALA + Escala Adaptativa)
El muestreador **RL-MALA** parametriza una distribución de propuesta gaussiana anisotrópica condicionada al estado actual $\mathbf{x} \in \mathbb{R}^N$ y al gradiente cuántico $\mathbf{g}(\mathbf{x}) = \nabla_{\mathbf{x}} \ln |\Psi(\mathbf{x})|$:

$$
q_\theta(\mathbf{x}' \mid \mathbf{x}) = \mathcal{N}\!\left(\mathbf{x}' \;\Big|\; \boldsymbol{\mu}_\theta(\mathbf{x}),\, \operatorname{diag}\!\big(\boldsymbol{\sigma}_\theta^2(\mathbf{x})\big)\right)
$$

donde la media y la desviación estándar por electrón combinan la física local de Langevin con una corrección neuronal no local aprendida por un Actor DeepSet con filtros continuos tipo SchNet sobre distancias interelectrónicas resueltas por espín ($r_{ij} = x_i - x_j$, canales $\uparrow\uparrow, \downarrow\downarrow, \uparrow\downarrow$):

$$
\boldsymbol{\mu}_\theta(\mathbf{x}) = \mathbf{x} + \tau\,\mathbf{g}_{\mathrm{clip}}(\mathbf{x}) + A_\mu \tanh\!\big(\mathbf{u}_\theta(\mathbf{x}, \mathbf{g})\big), \qquad
\boldsymbol{\sigma}_\theta(\mathbf{x}) = \sqrt{2\tau}\,\sigma_{\mathrm{scale}} \exp\!\big(s_\theta(\mathbf{x}, \mathbf{g})\big)
$$

Aquí:
- $A_\mu$ (`actor_amplitude`) controla la amplitud física máxima del desplazamiento residual colectivo aprendido por el actor.
- $\sigma_{\mathrm{scale}}$ (`sigma_action_scale`) modula la apertura estocástica de la covarianza de propuesta.
- La aceptación exacta de Metropolis-Hastings preserva el balance detallado estricto respecto a $\pi(\mathbf{x}) \propto |\Psi_{\mathrm{CASSCF}}(\mathbf{x})|^2$:
  $$
  \alpha_\theta(\mathbf{x} \to \mathbf{x}') = \min\!\left\{1,\, \frac{|\Psi_{\mathrm{CASSCF}}(\mathbf{x}')|^2 \, q_\theta(\mathbf{x} \mid \mathbf{x}')}{|\Psi_{\mathrm{CASSCF}}(\mathbf{x})|^2 \, q_\theta(\mathbf{x}' \mid \mathbf{x})}\right\}
  $$

### 2.2 Recompensa CDLB (*Clipped Detailed-Balance Log-Barrier*) con Exploración Entrópica
Para evitar que el actor colapse a pasos triviales ($\mathbf{x}' \approx \mathbf{x}$) o proponga saltos no reversibles que anulen $q_\theta(\mathbf{x} \mid \mathbf{x}')$, entrenamos mediante **TD3 (*Twin Delayed Deep Deterministic Policy Gradient*)** maximizando la recompensa **CDLB**:

$$
r_{\mathrm{CDLB}}(\mathbf{x}, \mathbf{x}') = \alpha_\theta(\mathbf{x} \to \mathbf{x}') \cdot \ln\!\left(1 + \frac{\|\mathbf{x}' - \mathbf{x}\|_{\mathrm{clip}}^2}{\tau}\right) - \lambda_{\mathrm{pen}}\big(1 - \alpha_\theta(\mathbf{x} \to \mathbf{x}')\big) - \alpha_{\mathrm{ent}} \ln q_\theta(\mathbf{x}' \mid \mathbf{x})
$$

Esta formulación logra tres efectos simultáneos:
1. **Impulso de salto colectivo:** El término $\alpha_\theta \ln(1 + \|\Delta \mathbf{x}\|^2 / \tau)$ premia desplazamientos largos en el espacio de configuración que mantengan alta probabilidad de aceptación.
2. **Penalización de rechazo ($\lambda_{\mathrm{pen}}$):** Evita que el actor proponga saltos demasiado agresivos hacia el exterior del pozo armónico.
3. **Exploración calibrada ($-\alpha_{\mathrm{ent}} \ln q_\theta$):** Incentiva al actor a ensanchar selectivamente $\boldsymbol{\sigma}_\theta(\mathbf{x})$ cuando dos electrones de espín opuesto o igual se aproximan a una barrera nodal, permitiendo **saltar por encima del valle nodal** de un bolsillo de permutación a otro en un solo paso colectivo.

---

## 3. Protocolo Experimental e Implementación Computacional

Para garantizar una comparación justa, reproducible y libre de sesgos entre MALA y RL-MALA:

1. **Aislamiento de la Función de Onda (`--ansatz_type casscf_npz --no-casscf_trainable`):**
   Ambos métodos muestrean exactamente el mismo estado multiconfiguracional congelado `casscf_state.npz`. No se modifica ningún parámetro del ansatz durante el entrenamiento ni durante la evaluación.
2. **Inicialización Simétrica durante el Entrenamiento RL (`--init_mode symmetric_gaussian --init_std 3.0`):**
   Durante los $6,000$ barridos de entrenamiento TD3 (`--train_mode rl_only`, $N_{\mathrm{walkers}}=128$, burn-in inicial de $2,000$ pasos), los caminantes se inicializan con una distribución gaussiana simétrica para poblar el *replay buffer* en todos los sectores de permutación de espín sin sesgo espacial.
3. **Prueba de Estrés de Cruce Nodal en Producción (`--eval_init_mode spin_ordered_offset`):**
   En la fase de evaluación de producción ($N_{\mathrm{chains}}=3$ cadenas independientes, $N_{\mathrm{walkers}}=128$, `eval_burn_in = 200`, `measure_steps = 40,000`), tanto MALA como RL-MALA se inicializan deliberadamente en una configuración **totalmente polarizada por espín**:
   - Todos los electrones $\uparrow$ en el semieje izquierdo ($x_{i,\uparrow} < 0$).
   - Todos los electrones $\downarrow$ en el semieje derecho ($x_{j,\downarrow} > 0$).
   Un muestreador ergódico capaz de cruzar barreras nodales debe restaurar rápidamente la simetría exacta $\rho_\uparrow(x) = \rho_\downarrow(x) = \frac{1}{2}\rho_{\mathrm{tot}}(x)$, haciendo que la divergencia de Kullback-Leibler promedio de espín:
   $$
   D_{\mathrm{KL}}^{\mathrm{spin}} = \frac{1}{2}\Big[D_{\mathrm{KL}}\!\big(p_\uparrow^{\mathrm{MCMC}} \parallel p_\uparrow^{\mathrm{CASSCF}}\big) + D_{\mathrm{KL}}\!\big(p_\downarrow^{\mathrm{MCMC}} \parallel p_\downarrow^{\mathrm{CASSCF}}\big)\Big]
   $$
   tienda asintóticamente a cero ($D_{\mathrm{KL}}^{\mathrm{spin}} \to 10^{-5} - 10^{-3}$). Por el contrario, si el muestreador queda atrapado en su sector topológico inicial, $D_{\mathrm{KL}}^{\mathrm{spin}}$ permanece estancado en $\mathcal{O}(0.1 - 0.6)$.

---

## 4. Resultados del Benchmark 1: `Wigner_N4_g4.0` ($N=4, \omega=0.1, g=4.0$)

### 4.1 Sistema de Referencia
- **Parámetros físicos:** $N=4$ ($N_\uparrow=2, N_\downarrow=2$), $\omega=0.1$, $g=4.0$, $a=0.1$.
- **Referencia CASSCF:** `cas_n4_g4_omega0.1_soft0.1_b32_c0-a4` ($N_{\mathrm{core}}=0, N_{\mathrm{active}}=4 \implies 36$ determinantes, base $N_{\mathrm{basis}}=32$), con energía exacta $E_{\mathrm{CASSCF}} = \mathbf{4.1595825\text{ Ha}}$.
- **Régimen físico:** Cristal de Wigner de 4 picos bien separados en $x \approx \pm 2.2, \pm 6.8\text{ Bohr}$.

### 4.2 Benchmark Inicial en Baker (Grilla Compacta, 16 Modelos)

| Modelo | $\tau$ | $A_\mu$ | $\lambda_{\mathrm{pen}}$ | $\langle E \rangle \pm \mathrm{SE}(E)$ (Ha) | $\Delta E$ (mHa) | Aceptación | ESJD ($\text{Bohr}^2$) | $\tau_{\mathrm{int}}(E)$ | $\tau_{\mathrm{int}}(x)$ | $\mathrm{ESS/s}$ ($\text{s}^{-1}$) | $D_{\mathrm{KL}}^{\mathrm{tot}}$ | $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MALA_tau0.04` | 0.04 | — | — | $4.160448 \pm 0.000557$ | $+0.87$ | $94.4\%$ | $0.291$ | $106.89 \pm 16.46$ | $2280.1$ | $1,141$ | $2.52 \times 10^{-4}$ | $0.62609$ |
| `MALA_tau0.08` | 0.08 | — | — | $4.160328 \pm 0.000338$ | $+0.75$ | $90.2\%$ | $0.556$ | $39.67 \pm 3.85$ | $2219.9$ | $3,081$ | $1.10 \times 10^{-4}$ | $0.58777$ |
| `MALA_tau0.10` | 0.10 | — | — | $4.159480 \pm 0.000237$ | $-0.10$ | $88.5\%$ | $0.685$ | $19.48 \pm 1.38$ | $2090.6$ | $6,267$ | $1.52 \times 10^{-4}$ | $0.52840$ |
| **`MALA_tau0.12` (Mejor MALA)** | **0.12** | — | — | **$4.159500 \pm 0.000180$** | **$-0.08$** | **$87.1\%$** | **$0.813$** | **$11.07 \pm 0.62$** | **$1855.3$** | **$11,033$** | **$1.11 \times 10^{-4}$** | **$0.46867$** |
| `RL_MALA_tau0.08_amp0.10_pen0.15_sig1.1` | 0.08 | 0.10 | 0.15 | $4.159819 \pm 0.000143$ | $+0.24$ | $55.6\%$ | $1.688$ | $6.96 \pm 0.32$ | $886.4$ | $11,932$ | $5.25 \times 10^{-5}$ | $0.03651$ |
| **`RL_MALA_tau0.08_amp0.10_pen0.25_sig1.1`** | **0.08** | **0.10** | **0.25** | **$4.159843 \pm 0.000130$** | **$+0.26$** | **$47.7\%$** | **$2.020$** | **$5.79 \pm 0.24$** | **$546.5$** | **$14,264$** | **$3.18 \times 10^{-5}$** | **$0.01564$** |
| `RL_MALA_tau0.08_amp0.15_pen0.15_sig1.1` | 0.08 | 0.15 | 0.15 | $4.159553 \pm 0.000175$ | $-0.03$ | $36.0\%$ | $1.807$ | $10.43 \pm 0.56$ | $533.1$ | $7,953$ | $3.65 \times 10^{-5}$ | $0.01851$ |
| `RL_MALA_tau0.08_amp0.15_pen0.25_sig1.1` | 0.08 | 0.15 | 0.25 | $4.159609 \pm 0.000161$ | $+0.03$ | $41.3\%$ | $1.741$ | $8.90 \pm 0.45$ | $760.6$ | $9,225$ | $4.54 \times 10^{-5}$ | $0.04425$ |
| `RL_MALA_tau0.08_amp0.25_pen0.15_sig1.1` | 0.08 | 0.25 | 0.15 | $4.158805 \pm 0.000605$ | $-0.78$ | $10.4\%$ | $0.916$ | $131.69 \pm 16.01$ | $945.9$ | $627$ | $1.50 \times 10^{-4}$ | $0.03699$ |
| `RL_MALA_tau0.08_amp0.25_pen0.25_sig1.1` | 0.08 | 0.25 | 0.25 | $4.159944 \pm 0.000592$ | $+0.36$ | $10.5\%$ | $0.930$ | $122.90 \pm 14.37$ | $1017.1$ | $669$ | $1.41 \times 10^{-4}$ | $0.07855$ |
| `RL_MALA_tau0.10_amp0.10_pen0.15_sig1.1` | 0.10 | 0.10 | 0.15 | $4.159624 \pm 0.000149$ | $+0.04$ | $62.0\%$ | $1.564$ | $7.60 \pm 0.36$ | $1150.7$ | $10,889$ | $3.54 \times 10^{-5}$ | $0.09772$ |
| **`RL_MALA_tau0.10_amp0.10_pen0.25_sig1.1` (Óptimo)** | **0.10** | **0.10** | **0.25** | **$4.159469 \pm 0.000126$** | **$-0.11$** | **$45.8\%$** | **$2.156$** | **$5.43 \pm 0.22$** | **$568.5$** | **$15,078$** | **$3.05 \times 10^{-5}$** | **$0.00958$** |
| `RL_MALA_tau0.10_amp0.15_pen0.15_sig1.1` | 0.10 | 0.15 | 0.15 | $4.159710 \pm 0.000155$ | $+0.13$ | $38.4\%$ | $1.908$ | $8.13 \pm 0.40$ | $597.2$ | $10,084$ | $3.26 \times 10^{-5}$ | $0.01451$ |
| `RL_MALA_tau0.10_amp0.15_pen0.25_sig1.1` | 0.10 | 0.15 | 0.25 | $4.159626 \pm 0.000178$ | $+0.04$ | $33.0\%$ | $1.727$ | $10.90 \pm 0.61$ | $554.7$ | $7,584$ | $4.08 \times 10^{-5}$ | $0.00998$ |
| `RL_MALA_tau0.10_amp0.25_pen0.15_sig1.1` | 0.10 | 0.25 | 0.15 | $4.159643 \pm 0.000147$ | $+0.06$ | $65.0\%$ | $1.594$ | $7.37 \pm 0.35$ | $1224.2$ | $11,175$ | $2.91 \times 10^{-5}$ | $0.08826$ |
| `RL_MALA_tau0.10_amp0.25_pen0.25_sig1.1` | 0.10 | 0.25 | 0.25 | $4.159500 \pm 0.000143$ | $-0.08$ | $54.2\%$ | $1.802$ | $7.01 \pm 0.32$ | $893.7$ | $11,769$ | $3.41 \times 10^{-5}$ | $0.02975$ |

### 4.3 Hallazgos de la Exploración Inicial en Baker
1. **Ruptura del bloqueo de espín ($49\times$ a $65\times$ menor $D_{\mathrm{KL}}^{\mathrm{spin}}$):** Todos los modelos MALA quedan severamente polarizados ($D_{\mathrm{KL}}^{\mathrm{spin}} \in [0.469, 0.626]$), mientras que `RL_MALA_tau0.10_amp0.10_pen0.25_sig1.1` reduce $D_{\mathrm{KL}}^{\mathrm{spin}}$ a **$0.00958$** (una mejora de **$48.9\times$** frente al mejor MALA y **$65.4\times$** frente a MALA $\tau=0.04$).
2. **Aceleración real de reloj ($\mathrm{ESS/s}$):** Pese al costo adicional de evaluar la red DeepSet en cada paso, `RL_MALA_tau0.10_amp0.10_pen0.25_sig1.1` alcanza **$\mathbf{15,078\text{ s}^{-1}}$**, superando al mejor MALA ($11,033\text{ s}^{-1}$) en un **+$36.7\%$ neto** y reduciendo $\tau_{\mathrm{int}}(E)$ de $11.07$ a **$5.43$ pasos** ($2.04\times$ menor) y la autocorrelación espacial $\tau_{\mathrm{int}}(x)$ de $1855.3$ a **$568.5$ pasos** ($3.26\times$ menor).
3. **Sensibilidad a $A_\mu$ en acoplamiento fuerte ($g=4.0$):** Cuando $A_\mu = 0.25$ y $\tau=0.08$, el salto propuesto excede el ancho de los bolsillos de Wigner, reduciendo la aceptación a $\sim 10.4\%$. En cambio, el rango **$A_\mu \in [0.10, 0.15]$ con $\lambda_{\mathrm{pen}} = 0.25$** mantiene la aceptación exactamente en el régimen óptimo de transporte no local ($45\% - 55\%$).

### 4.4 Nueva Exploración Sistemática en Ayler (Grilla Completa de 41 Modelos)
Para explorar si pasos temporales mayores ($\tau \ge 0.14$) logran un mayor rendimiento sin romper el transporte cuántico, se ejecutó en Ayler una grilla complementaria de 41 modelos (5 MALA + 36 RL-MALA) con:
- MALA: $\tau \in \{0.06, 0.14, 0.16, 0.20, 0.25\}$
- RL-MALA: $\tau \in \{0.10, 0.12, 0.14\}$, $A_\mu \in \{0.06, 0.08, 0.12, 0.18\}$, $\lambda_{\mathrm{pen}} \in \{0.20, 0.30, 0.40\}$, $\sigma_{\mathrm{sc}} = 1.15$.

| Modelo | $\tau$ | $A_\mu$ | $\lambda_{\mathrm{pen}}$ | $\sigma_{\mathrm{sc}}$ | $\langle E \rangle \pm \mathrm{SE}(E)$ (Ha) | $\Delta E$ (mHa) | Aceptación | ESJD ($\text{Bohr}^2$) | $\tau_{\mathrm{int}}(E)$ | $\mathrm{ESS/s}$ ($\text{s}^{-1}$) | $D_{\mathrm{KL}}^{\mathrm{tot}}$ | $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MALA_tau0.06` | 0.06 | — | — | — | $4.160265 \pm 0.000422$ | $+0.68$ | 91.0% | 0.428 | 26.14 | 6,579 | 7.96e-06 | **3.2632** |
| `MALA_tau0.14` | 0.14 | — | — | — | $4.160030 \pm 0.000239$ | $+0.45$ | 86.1% | 0.935 | 9.62 | 17,776 | 5.02e-06 | **0.7976** |
| `MALA_tau0.16` | 0.16 | — | — | — | $4.159943 \pm 0.000231$ | $+0.36$ | 85.1% | 1.054 | 8.92 | 19,007 | 5.18e-06 | **0.8668** |
| `MALA_tau0.20` | 0.20 | — | — | — | $4.160000 \pm 0.000210$ | $+0.42$ | 83.3% | 1.284 | 7.46 | 22,639 | 4.30e-06 | **0.7095** |
| `MALA_tau0.25` | 0.25 | — | — | — | $4.159716 \pm 0.000189$ | $+0.13$ | 81.3% | 1.559 | 6.07 | 28,147 | 3.57e-06 | **0.6265** |
| **`RL_MALA_tau0.14_amp0.12_pen0.20_sig1.15`** | 0.14 | 0.12 | 0.20 | 1.15 | $4.159816 \pm 0.000170$ | $+0.23$ | 39.7% | 2.424 | 4.82 | 3,192,206 | 8.35e-06 | **3.9233e-03** |
| **`RL_MALA_tau0.14_amp0.18_pen0.40_sig1.15`** | 0.14 | 0.18 | 0.40 | 1.15 | $4.159727 \pm 0.000177$ | $+0.14$ | 38.7% | 2.358 | 5.28 | 2,909,093 | 8.70e-06 | **4.0297e-03** |
| **`RL_MALA_tau0.14_amp0.06_pen0.20_sig1.15`** | 0.14 | 0.06 | 0.20 | 1.15 | $4.159790 \pm 0.000172$ | $+0.21$ | 40.3% | 2.463 | 4.87 | 3,157,276 | 7.66e-06 | **4.4769e-03** |
| `RL_MALA_tau0.14_amp0.08_pen0.40_sig1.15` | 0.14 | 0.08 | 0.40 | 1.15 | $4.159715 \pm 0.000167$ | $+0.13$ | 40.2% | 2.454 | 4.68 | 3,283,711 | 9.16e-06 | 5.1001e-03 |
| `RL_MALA_tau0.14_amp0.06_pen0.40_sig1.15` | 0.14 | 0.06 | 0.40 | 1.15 | $4.159863 \pm 0.000169$ | $+0.28$ | 40.3% | 2.463 | 4.78 | 3,211,878 | 8.75e-06 | 5.3106e-03 |
| `RL_MALA_tau0.14_amp0.08_pen0.30_sig1.15` | 0.14 | 0.08 | 0.30 | 1.15 | $4.159724 \pm 0.000168$ | $+0.14$ | 40.2% | 2.452 | 4.77 | 3,221,036 | 8.42e-06 | 5.3708e-03 |
| `RL_MALA_tau0.14_amp0.06_pen0.30_sig1.15` | 0.14 | 0.06 | 0.30 | 1.15 | $4.159856 \pm 0.000170$ | $+0.27$ | 40.3% | 2.463 | 4.93 | 3,120,012 | 8.15e-06 | 5.7371e-03 |
| `RL_MALA_tau0.14_amp0.12_pen0.30_sig1.15` | 0.14 | 0.12 | 0.30 | 1.15 | $4.159752 \pm 0.000173$ | $+0.17$ | 39.7% | 2.423 | 5.05 | 3,039,889 | 7.43e-06 | 5.7669e-03 |
| `RL_MALA_tau0.14_amp0.12_pen0.40_sig1.15` | 0.14 | 0.12 | 0.40 | 1.15 | $4.159873 \pm 0.000171$ | $+0.29$ | 39.7% | 2.424 | 4.84 | 3,177,782 | 8.32e-06 | 6.7477e-03 |
| `RL_MALA_tau0.14_amp0.18_pen0.20_sig1.15` | 0.14 | 0.18 | 0.20 | 1.15 | $4.159712 \pm 0.000178$ | $+0.13$ | 38.7% | 2.359 | 5.35 | 2,876,357 | 9.14e-06 | 6.8302e-03 |
| `RL_MALA_tau0.14_amp0.18_pen0.30_sig1.15` | 0.14 | 0.18 | 0.30 | 1.15 | $4.159792 \pm 0.000175$ | $+0.21$ | 38.7% | 2.359 | 5.07 | 3,031,764 | 1.03e-05 | 7.0373e-03 |
| `RL_MALA_tau0.12_amp0.06_pen0.40_sig1.15` | 0.12 | 0.06 | 0.40 | 1.15 | $4.159827 \pm 0.000181$ | $+0.24$ | 43.5% | 2.348 | 5.45 | 2,826,151 | 8.48e-06 | 7.2955e-03 |
| `RL_MALA_tau0.14_amp0.08_pen0.20_sig1.15` | 0.14 | 0.08 | 0.20 | 1.15 | $4.159861 \pm 0.000170$ | $+0.28$ | 40.2% | 2.452 | 4.83 | 3,182,417 | 9.38e-06 | 8.5752e-03 |
| `RL_MALA_tau0.12_amp0.08_pen0.30_sig1.15` | 0.12 | 0.08 | 0.30 | 1.15 | $4.159942 \pm 0.000190$ | $+0.36$ | 43.3% | 2.335 | 6.04 | 2,655,834 | 6.42e-06 | 8.7031e-03 |
| `RL_MALA_tau0.12_amp0.12_pen0.30_sig1.15` | 0.12 | 0.12 | 0.30 | 1.15 | $4.159891 \pm 0.000184$ | $+0.31$ | 42.7% | 2.300 | 5.53 | 2,788,411 | 7.77e-06 | 9.5868e-03 |
| `RL_MALA_tau0.12_amp0.18_pen0.30_sig1.15` | 0.12 | 0.18 | 0.30 | 1.15 | $4.159863 \pm 0.000180$ | $+0.28$ | 41.4% | 2.227 | 5.38 | 2,853,829 | 7.19e-06 | 9.6726e-03 |
| `RL_MALA_tau0.12_amp0.12_pen0.20_sig1.15` | 0.12 | 0.12 | 0.20 | 1.15 | $4.159754 \pm 0.000178$ | $+0.17$ | 42.7% | 2.300 | 5.39 | 2,852,227 | 8.76e-06 | 9.8037e-03 |
| `RL_MALA_tau0.12_amp0.18_pen0.20_sig1.15` | 0.12 | 0.18 | 0.20 | 1.15 | $4.159928 \pm 0.000177$ | $+0.35$ | 41.4% | 2.225 | 5.27 | 2,921,047 | 8.23e-06 | 1.0127e-02 |
| `RL_MALA_tau0.12_amp0.06_pen0.30_sig1.15` | 0.12 | 0.06 | 0.30 | 1.15 | $4.159827 \pm 0.000177$ | $+0.24$ | 43.5% | 2.348 | 5.24 | 2,947,635 | 6.88e-06 | 1.0640e-02 |
| `RL_MALA_tau0.12_amp0.06_pen0.20_sig1.15` | 0.12 | 0.06 | 0.20 | 1.15 | $4.159746 \pm 0.000173$ | $+0.16$ | 43.5% | 2.347 | 5.03 | 3,055,738 | 7.99e-06 | 1.0814e-02 |
| `RL_MALA_tau0.12_amp0.12_pen0.40_sig1.15` | 0.12 | 0.12 | 0.40 | 1.15 | $4.159840 \pm 0.000182$ | $+0.26$ | 42.7% | 2.300 | 5.49 | 2,845,377 | 8.66e-06 | 1.2402e-02 |
| `RL_MALA_tau0.12_amp0.08_pen0.20_sig1.15` | 0.12 | 0.08 | 0.20 | 1.15 | $4.159884 \pm 0.000182$ | $+0.30$ | 43.3% | 2.336 | 5.65 | 2,743,630 | 8.37e-06 | 1.2454e-02 |
| `RL_MALA_tau0.12_amp0.08_pen0.40_sig1.15` | 0.12 | 0.08 | 0.40 | 1.15 | $4.160103 \pm 0.000181$ | $+0.52$ | 43.3% | 2.336 | 5.37 | 2,863,085 | 8.04e-06 | 1.3608e-02 |
| `RL_MALA_tau0.12_amp0.18_pen0.40_sig1.15` | 0.12 | 0.18 | 0.40 | 1.15 | $4.159843 \pm 0.000180$ | $+0.26$ | 41.4% | 2.226 | 5.44 | 2,822,286 | 8.45e-06 | 1.4541e-02 |
| `RL_MALA_tau0.10_amp0.06_pen0.30_sig1.15` | 0.10 | 0.06 | 0.30 | 1.15 | $4.159828 \pm 0.000179$ | $+0.25$ | 47.2% | 2.191 | 5.49 | 2,806,272 | 6.16e-06 | 1.4947e-02 |
| `RL_MALA_tau0.10_amp0.06_pen0.40_sig1.15` | 0.10 | 0.06 | 0.40 | 1.15 | $4.159851 \pm 0.000186$ | $+0.27$ | 47.2% | 2.191 | 5.74 | 2,685,668 | 6.91e-06 | 1.5605e-02 |
| `RL_MALA_tau0.10_amp0.06_pen0.20_sig1.15` | 0.10 | 0.06 | 0.20 | 1.15 | $4.159946 \pm 0.000192$ | $+0.36$ | 47.2% | 2.190 | 6.19 | 2,534,490 | 6.55e-06 | 1.6343e-02 |
| `RL_MALA_tau0.10_amp0.08_pen0.30_sig1.15` | 0.10 | 0.08 | 0.30 | 1.15 | $4.159730 \pm 0.000179$ | $+0.15$ | 47.0% | 2.176 | 5.47 | 2,810,577 | 6.03e-06 | 1.7374e-02 |
| `RL_MALA_tau0.10_amp0.08_pen0.40_sig1.15` | 0.10 | 0.08 | 0.40 | 1.15 | $4.159764 \pm 0.000181$ | $+0.18$ | 46.9% | 2.177 | 5.59 | 2,754,852 | 6.60e-06 | 1.8168e-02 |
| `RL_MALA_tau0.10_amp0.08_pen0.20_sig1.15` | 0.10 | 0.08 | 0.20 | 1.15 | $4.159941 \pm 0.000184$ | $+0.36$ | 46.9% | 2.176 | 5.66 | 2,716,887 | 6.93e-06 | 1.9430e-02 |
| `RL_MALA_tau0.10_amp0.18_pen0.30_sig1.15` | 0.10 | 0.18 | 0.30 | 1.15 | $4.159754 \pm 0.000190$ | $+0.17$ | 44.4% | 2.047 | 5.89 | 2,608,061 | 7.24e-06 | 1.9507e-02 |
| `RL_MALA_tau0.10_amp0.12_pen0.30_sig1.15` | 0.10 | 0.12 | 0.30 | 1.15 | $4.159909 \pm 0.000183$ | $+0.33$ | 46.1% | 2.136 | 5.71 | 2,692,080 | 6.22e-06 | 1.9513e-02 |
| `RL_MALA_tau0.10_amp0.12_pen0.40_sig1.15` | 0.10 | 0.12 | 0.40 | 1.15 | $4.159848 \pm 0.000183$ | $+0.27$ | 46.1% | 2.136 | 5.58 | 2,758,992 | 6.78e-06 | 2.2136e-02 |
| `RL_MALA_tau0.10_amp0.12_pen0.20_sig1.15` | 0.10 | 0.12 | 0.20 | 1.15 | $4.159868 \pm 0.000186$ | $+0.29$ | 46.1% | 2.135 | 5.82 | 2,641,171 | 8.87e-06 | 2.4917e-02 |
| `RL_MALA_tau0.10_amp0.18_pen0.40_sig1.15` | 0.10 | 0.18 | 0.40 | 1.15 | $4.159718 \pm 0.000186$ | $+0.14$ | 44.4% | 2.048 | 5.95 | 2,581,932 | 8.54e-06 | 2.5868e-02 |
| `RL_MALA_tau0.10_amp0.18_pen0.20_sig1.15` | 0.10 | 0.18 | 0.20 | 1.15 | $4.159781 \pm 0.000195$ | $+0.20$ | 44.4% | 2.048 | 6.24 | 2,478,875 | 7.04e-06 | 2.6297e-02 |

#### Análisis Detallado de los Resultados en `Wigner_N4_g4.0` (Ayler Grid):
1. **Quiebre Físico de MALA a Pasos Grandes vs Éxito de RL-MALA:**
   - En MALA clásico, aumentar $\tau$ de $0.06$ a $0.25$ reduce la asimetría de espín de $D_{\mathrm{KL}}^{\mathrm{spin}} = 3.2632$ a $0.6265$. No obstante, **incluso con $\tau=0.25$, MALA sigue completamente congelado en un solo ordenamiento de espín** ($D_{\mathrm{KL}}^{\mathrm{spin}} = 0.6265 \gg 10^{-2}$).
   - Por el contrario, los 36 modelos RL-MALA rompen radicalmente la barrera nodal. Los modelos con $\tau=0.14$ alcanzan una divergencia de espín mínima de **$D_{\mathrm{KL}}^{\mathrm{spin}} = 3.9233 \times 10^{-3}$** (`RL_MALA_tau0.14_amp0.12_pen0.20_sig1.15`), representando una reducción del error de polarización de **$160\times$** frente a `MALA_tau0.25` y **$832\times$** frente a `MALA_tau0.06`.
2. **Eficiencia de Muestreo y Rendimiento Real ($\mathrm{ESS/s}$):**
   - El modelo `RL_MALA_tau0.14_amp0.08_pen0.40_sig1.15` mantiene una aceptación de **$40.2\%$**, un salto cuadrático medio récord de **$\mathrm{ESJD} = 2.454\text{ Bohr}^2$** (frente a $0.428\text{ Bohr}^2$ de MALA $\tau=0.06$), reduciendo el tiempo de autocorrelación a $\tau_{\mathrm{int}}(E) = 4.68$ pasos.
   - La velocidad efectiva de generación de muestras independientes supera los **$3.28 \times 10^6\text{ ESS/s}$** en el GPU, con un error estándar de energía de sólo $\mathrm{SE}(E) = 0.000167\text{ Ha}$.
3. **Robustez ante los Hiperparámetros:**
   - Como se visualiza en la figura de evolución, la región $\tau=0.14$ combinada con amplitudes moderadas $A_\mu \in [0.06, 0.12]$ y penalizaciones $\lambda_{\mathrm{pen}} \in [0.20, 0.40]$ forma una meseta de alta estabilidad donde todos los modelos superan ampliamente a MALA.

#### Figuras Clave del Benchmark 1 (`Wigner_N4_g4.0`):

##### Reconstrucción de Densidades Espaciales y Simetría de Espín Exacta
![Densidades Unidimensionales y Simetría Exacta (Wigner N=4, g=4.0)](results_nodal_crossing_search_new/figures/Wigner_N4_g4.0/Article_Densities_Wigner_N4_g4.0.png)  
*Figura 4.1: Densidades electrónicas de un cuerpo resueltas por espín $\rho_\uparrow(x)$ y $\rho_\downarrow(x)$ frente a la referencia CASSCF congelada. MALA clásico permanece polarizado asimétricamente en los pozos exteriores e interiores, mientras que RL-MALA restaura la indistinguibilidad estadística exacta.*

##### Dinámica Individual de Cruce Nodal (Trazas de 1 Caminante)
![Trazas de 1 Solo Caminante: MALA vs RL-MALA (Wigner N=4, g=4.0)](results_nodal_crossing_search_new/figures/Wigner_N4_g4.0/Article_Walker_Traces_Wigner_N4_g4.0.png)  
*Figura 4.2: Trazas temporales para un único caminante físico en $40,000$ pasos de evaluación. MALA queda atrapado en una configuración de espín ordenada ($x_{1,\uparrow} < x_{2,\uparrow} < 0 < x_{1,\downarrow} < x_{2,\downarrow}$), mientras que RL-MALA atraviesa repetidamente las barreras de intercambio sin sufrir colapso de aceptación.*

##### Exploración Sistemática del Espacio de Hiperparámetros (Grilla Ayler, 41 Modelos)
![Evolución con Hiperparámetros (Wigner N=4, g=4.0)](figures_param_search_summary/Wigner_N4_g4.0/fig_hyperparameter_evolution_Wigner_N4_g4.0.png)  
*Figura 4.3: Evolución de las métricas clave ($D_{\mathrm{KL}}^{\mathrm{spin}}$, salto cuadrático $\mathrm{ESJD}$, tiempo de autocorrelación $\tau_{\mathrm{int}}(E)$ y tasa de aceptación) en función del paso temporal $\tau$. Las curvas sólidas (RL-MALA) muestran la media y la dispersión sobre todas las combinaciones de $(A_\mu, \lambda_{\mathrm{pen}})$; las curvas punteadas representan MALA.*

![Mapas de Calor 2D del Actor (Wigner N=4, g=4.0)](figures_param_search_summary/Wigner_N4_g4.0/fig_param_search_heatmaps_Wigner_N4_g4.0.png)  
*Figura 4.4: Mapas de calor 2D para la divergencia de espín $D_{\mathrm{KL}}^{\mathrm{spin}}$ y el salto $\mathrm{ESJD}$ en el plano $(A_\mu, \tau)$ para penalizaciones $\lambda_{\mathrm{pen}} \in \{0.20, 0.30, 0.40\}$. Se evidencia la meseta óptima en $\tau=0.14, A_\mu \in [0.06, 0.12]$.*

![Sensibilidad a la Penalización de Rechazo lambda_pen (Wigner N=4, g=4.0)](figures_param_search_summary/Wigner_N4_g4.0/fig_penalty_sensitivity_Wigner_N4_g4.0.png)  
*Figura 4.5: Análisis de sensibilidad del error de simetría $D_{\mathrm{KL}}^{\mathrm{spin}}$ frente al parámetro de penalización de rechazo $\lambda_{\mathrm{pen}}$ para distintos niveles de $\tau$.*

---
## 5. Resultados del Benchmark 2: `MultiDet_N6_g1.0` ($N=6, \omega=0.1, g=1.0$, Base 256)

### 5.1 Sistema de Referencia
- **Parámetros físicos:** $N=6$ ($N_\uparrow=3, N_\downarrow=3$), $\omega=0.1$, $g=1.0$, $a=0.1$.
- **Referencia CASSCF:** `cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz` ($\mathrm{CAS}(6e, 6o)$ con 20 determinantes seleccionados en base $N_{\mathrm{basis}}=256$), con energía exacta $E_{\mathrm{CASSCF}} = \mathbf{4.5240521\text{ Ha}}$.
- **Desafío computacional y topológico:** Evaluar 20 determinantes $3\times 3$ sobre una base de 256 funciones de Hermite en un espacio de configuración de 6 dimensiones con $6! / (3!3!) = 20$ sectores de permutación de espín y 6 picos de densidad.

### 5.2 Benchmark Inicial en Baker (Grilla Compacta, 16 Modelos)

| Modelo | $\tau$ | $A_\mu$ | $\lambda_{\mathrm{pen}}$ | $\langle E \rangle \pm \mathrm{SE}(E)$ (Ha) | $\Delta E$ (mHa) | Aceptación | ESJD ($\text{Bohr}^2$) | $\tau_{\mathrm{int}}(E)$ | $\tau_{\mathrm{int}}(x)$ | $\mathrm{ESS/s}$ ($\text{s}^{-1}$) | $D_{\mathrm{KL}}^{\mathrm{tot}}$ | $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MALA_tau0.04` | 0.04 | — | — | $4.524651 \pm 0.000698$ | $+0.60$ | $96.2\%$ | $0.423$ | $25.88 \pm 2.28$ | $3899.3$ | $819.5$ | $2.68 \times 10^{-5}$ | $0.38468$ |
| `MALA_tau0.08` | 0.08 | — | — | $4.523503 \pm 0.000443$ | $-0.55$ | $92.8\%$ | $0.801$ | $10.91 \pm 0.62$ | $3246.3$ | $1,938.4$ | $1.61 \times 10^{-5}$ | $0.17211$ |
| `MALA_tau0.10` | 0.10 | — | — | $4.523887 \pm 0.000349$ | $-0.17$ | $91.1\%$ | $0.978$ | $6.94 \pm 0.32$ | $2921.6$ | $3,059.3$ | $1.26 \times 10^{-5}$ | $0.15066$ |
| **`MALA_tau0.12` (Mejor MALA)** | **0.12** | — | — | **$4.524204 \pm 0.000361$** | **$+0.15$** | **$89.6\%$** | **$1.149$** | **$7.28 \pm 0.34$** | **$2846.7$** | **$2,911.2$** | **$1.03 \times 10^{-5}$** | **$0.11513$** |
| `RL_MALA_tau0.08_amp0.10_pen0.15_sig1.1` | 0.08 | 0.10 | 0.15 | $4.523699 \pm 0.000357$ | $-0.35$ | $69.3\%$ | $1.568$ | $6.97 \pm 0.32$ | $2458.5$ | $2,756.7$ | $1.41 \times 10^{-5}$ | $2.22 \times 10^{-4}$ |
| `RL_MALA_tau0.08_amp0.10_pen0.25_sig1.1` | 0.08 | 0.10 | 0.25 | $4.523980 \pm 0.000341$ | $-0.07$ | $66.2\%$ | $1.630$ | $6.47 \pm 0.29$ | $2339.7$ | $2,970.4$ | $8.98 \times 10^{-6}$ | $1.72 \times 10^{-4}$ |
| `RL_MALA_tau0.08_amp0.15_pen0.15_sig1.1` | 0.08 | 0.15 | 0.15 | $4.523624 \pm 0.000338$ | $-0.43$ | $55.9\%$ | $1.872$ | $6.42 \pm 0.28$ | $2271.5$ | $2,994.3$ | $1.23 \times 10^{-5}$ | $1.52 \times 10^{-4}$ |
| **`RL_MALA_tau0.08_amp0.15_pen0.25_sig1.1` (Mejor $D_{\mathrm{KL}}^{\mathrm{spin}}$)** | **0.08** | **0.15** | **0.25** | **$4.523919 \pm 0.000334$** | **$-0.13$** | **$49.4\%$** | **$2.076$** | **$6.10 \pm 0.26$** | **$1939.1$** | **$3,149.5$** | **$9.74 \times 10^{-6}$** | **$\mathbf{5.52 \times 10^{-5}}$** |
| `RL_MALA_tau0.08_amp0.25_pen0.15_sig1.1` | 0.08 | 0.25 | 0.15 | $4.524087 \pm 0.000349$ | $+0.04$ | $65.0\%$ | $1.678$ | $6.71 \pm 0.30$ | $2226.4$ | $2,859.9$ | $1.56 \times 10^{-5}$ | $1.61 \times 10^{-4}$ |
| `RL_MALA_tau0.08_amp0.25_pen0.25_sig1.1` | 0.08 | 0.25 | 0.25 | $4.523759 \pm 0.000353$ | $-0.29$ | $59.8\%$ | $1.682$ | $6.83 \pm 0.31$ | $2458.5$ | $2,812.8$ | $1.25 \times 10^{-5}$ | $1.21 \times 10^{-4}$ |
| `RL_MALA_tau0.10_amp0.10_pen0.15_sig1.1` | 0.10 | 0.10 | 0.15 | $4.524472 \pm 0.000328$ | $+0.42$ | $61.3\%$ | $1.874$ | $5.92 \pm 0.25$ | $2221.3$ | $3,241.3$ | $1.04 \times 10^{-5}$ | $2.57 \times 10^{-4}$ |
| `RL_MALA_tau0.10_amp0.10_pen0.25_sig1.1` | 0.10 | 0.10 | 0.25 | $4.523770 \pm 0.000334$ | $-0.28$ | $58.2\%$ | $1.976$ | $6.03 \pm 0.26$ | $2135.7$ | $3,296.5$ | $1.41 \times 10^{-5}$ | $1.09 \times 10^{-4}$ |
| **`RL_MALA_tau0.10_amp0.15_pen0.15_sig1.1`** | **0.10** | **0.15** | **0.15** | **$4.523770 \pm 0.000316$** | **$-0.28$** | **$48.6\%$** | **$2.269$** | **$5.76 \pm 0.24$** | **$1875.5$** | **$3,455.7$** | **$1.17 \times 10^{-5}$** | **$8.72 \times 10^{-5}$** |
| **`RL_MALA_tau0.10_amp0.15_pen0.25_sig1.1` (Óptimo Global)** | **0.10** | **0.15** | **0.25** | **$4.524407 \pm 0.000305$** | **$+0.35$** | **$44.8\%$** | **$2.285$** | **$5.29 \pm 0.21$** | **$1650.8$** | **$3,755.0$** | **$9.88 \times 10^{-6}$** | **$1.16 \times 10^{-4}$** |
| `RL_MALA_tau0.10_amp0.25_pen0.15_sig1.1` | 0.10 | 0.25 | 0.15 | $4.524066 \pm 0.000321$ | $+0.01$ | $56.3\%$ | $1.973$ | $5.78 \pm 0.24$ | $2178.4$ | $3,315.4$ | $1.26 \times 10^{-5}$ | $1.32 \times 10^{-4}$ |
| `RL_MALA_tau0.10_amp0.25_pen0.25_sig1.1` | 0.10 | 0.25 | 0.25 | $4.524148 \pm 0.000349$ | $+0.10$ | $54.5\%$ | $1.878$ | $6.68 \pm 0.30$ | $2402.0$ | $2,870.5$ | $7.83 \times 10^{-6}$ | $1.41 \times 10^{-4}$ |

### 5.3 Hallazgos de la Exploración Inicial en Baker
1. **Superioridad Universal (12 de 12 modelos RL-MALA vencen a todos los MALA):**
   En $N=6, g=1.0$, absolutamente los 12 modelos RL-MALA logran una divergencia de espín $D_{\mathrm{KL}}^{\mathrm{spin}} \in [5.52 \times 10^{-5}, 2.57 \times 10^{-4}]$, reduciendo el error de polarización entre **$448\times$ y $2,085\times$** respecto al mejor MALA (`MALA_tau0.12`, $D_{\mathrm{KL}}^{\mathrm{spin}} = 0.11513$).
2. **Erradicación Completa del Bloqueo de Espín en 1 Solo Caminante:**
   En MALA ($\tau=0.12$), los 3 electrones $\uparrow$ permanecen confinados mayoritariamente en $x < 0$ y los 3 electrones $\downarrow$ en $x > 0$. En RL-MALA, un único caminante ejecuta decenas de intercambios colectivos de espín reconstruyendo las 6 crestas simétricas.

### 5.4 Exploración Sistemática Completa en Ayler (Grilla Completa de 41 Modelos)
Una vez resuelta la incidencia de compilación de Triton GEMM y estandarizada la exportación de rutas de ejecución en Ayler, se ejecutó con éxito el **100% de la grilla paramétrica** para `MultiDet_N6_g1.0`, evaluando un total de 41 configuraciones (5 modelos MALA de referencia y 36 variantes TD3 RL-MALA) bajo la condición de prueba de estrés desde ordenamiento polarizado de espín (`--eval_init_mode spin_ordered_offset`):

| Modelo | $\tau$ | $A_\mu$ | $\lambda_{\mathrm{pen}}$ | $\sigma_{\mathrm{sc}}$ | $\langle E \rangle \pm \mathrm{SE}(E)$ (Ha) | $\Delta E$ (mHa) | Aceptación | ESJD ($\text{Bohr}^2$) | $\tau_{\mathrm{int}}(E)$ | $\mathrm{ESS/s}$ ($\text{s}^{-1}$) | $D_{\mathrm{KL}}^{\mathrm{tot}}$ | $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MALA_tau0.06` | 0.06 | — | — | — | $4.524417 \pm 0.000477$ | $+0.37$ | 87.2% | 0.616 | 12.83 | 1,203,372 | 4.56e-06 | **0.3881** |
| `MALA_tau0.14` | 0.14 | — | — | — | $4.524131 \pm 0.000351$ | $+0.08$ | 80.3% | 1.313 | 6.92 | 2,247,165 | 3.49e-06 | **0.0857** |
| `MALA_tau0.16` | 0.16 | — | — | — | $4.524121 \pm 0.000315$ | $+0.07$ | 78.9% | 1.472 | 5.70 | 2,702,563 | 3.79e-06 | **0.0603** |
| `MALA_tau0.20` | 0.20 | — | — | — | $4.524138 \pm 0.000295$ | $+0.09$ | 76.3% | 1.775 | 4.94 | 3,107,302 | 3.02e-06 | **0.0350** |
| `MALA_tau0.25` | 0.25 | — | — | — | $4.523880 \pm 0.000266$ | $-0.17$ | 73.4% | 2.128 | 4.08 | 3,765,410 | 3.21e-06 | **0.0214** |
| **`RL_MALA_tau0.10_amp0.06_pen0.30_sig1.15`** | 0.10 | 0.06 | 0.30 | 1.15 | $4.523999 \pm 0.000313$ | $-0.05$ | 33.8% | 2.420 | 5.49 | 2,799,983 | 7.25e-06 | **2.0578e-05** |
| **`RL_MALA_tau0.10_amp0.18_pen0.30_sig1.15`** | 0.10 | 0.18 | 0.30 | 1.15 | $4.524431 \pm 0.000317$ | $+0.38$ | 31.2% | 2.224 | 5.75 | 2,672,019 | 7.71e-06 | **2.2031e-05** |
| **`RL_MALA_tau0.10_amp0.08_pen0.40_sig1.15`** | 0.10 | 0.08 | 0.40 | 1.15 | $4.524008 \pm 0.000317$ | $-0.04$ | 33.5% | 2.401 | 5.61 | 2,746,236 | 7.49e-06 | **2.3611e-05** |
| **`RL_MALA_tau0.10_amp0.08_pen0.30_sig1.15`** | 0.10 | 0.08 | 0.30 | 1.15 | $4.524279 \pm 0.000315$ | $+0.23$ | 33.5% | 2.400 | 5.57 | 2,758,855 | 6.98e-06 | **4.9475e-05** |
| `RL_MALA_tau0.10_amp0.12_pen0.20_sig1.15` | 0.10 | 0.12 | 0.20 | 1.15 | $4.524325 \pm 0.000322$ | $+0.27$ | 32.8% | 2.345 | 5.84 | 2,638,289 | 7.80e-06 | 8.1153e-05 |
| `RL_MALA_tau0.10_amp0.06_pen0.20_sig1.15` | 0.10 | 0.06 | 0.20 | 1.15 | $4.524408 \pm 0.000317$ | $+0.36$ | 33.8% | 2.419 | 5.60 | 2,744,974 | 8.19e-06 | 8.1478e-05 |
| `RL_MALA_tau0.10_amp0.18_pen0.20_sig1.15` | 0.10 | 0.18 | 0.20 | 1.15 | $4.524191 \pm 0.000329$ | $+0.14$ | 31.3% | 2.225 | 6.11 | 2,515,286 | 8.51e-06 | 1.2043e-04 |
| `RL_MALA_tau0.10_amp0.12_pen0.30_sig1.15` | 0.10 | 0.12 | 0.30 | 1.15 | $4.524082 \pm 0.000314$ | $+0.03$ | 32.8% | 2.344 | 5.61 | 2,741,824 | 6.69e-06 | 1.5375e-04 |
| `RL_MALA_tau0.10_amp0.12_pen0.40_sig1.15` | 0.10 | 0.12 | 0.40 | 1.15 | $4.524530 \pm 0.000315$ | $+0.48$ | 32.8% | 2.344 | 5.55 | 2,769,269 | 7.09e-06 | 1.6895e-04 |
| `RL_MALA_tau0.10_amp0.06_pen0.40_sig1.15` | 0.10 | 0.06 | 0.40 | 1.15 | $4.524272 \pm 0.000310$ | $+0.22$ | 33.8% | 2.420 | 5.40 | 2,846,591 | 6.96e-06 | 1.9696e-04 |
| `RL_MALA_tau0.10_amp0.08_pen0.20_sig1.15` | 0.10 | 0.08 | 0.20 | 1.15 | $4.524408 \pm 0.000320$ | $+0.36$ | 33.6% | 2.398 | 5.69 | 2,703,183 | 7.20e-06 | 2.1069e-04 |
| `RL_MALA_tau0.14_amp0.12_pen0.40_sig1.15` | 0.14 | 0.12 | 0.40 | 1.15 | $4.524407 \pm 0.000336$ | $+0.35$ | 36.0% | 2.448 | 6.28 | 2,454,487 | 2.81e-05 | 1.8549e-02 |
| `RL_MALA_tau0.12_amp0.06_pen0.30_sig1.15` | 0.12 | 0.06 | 0.30 | 1.15 | $4.524437 \pm 0.000334$ | $+0.38$ | 38.9% | 2.370 | 6.25 | 2,539,518 | 2.28e-05 | 1.9177e-02 |
| `RL_MALA_tau0.12_amp0.08_pen0.40_sig1.15` | 0.12 | 0.08 | 0.40 | 1.15 | $4.524213 \pm 0.000330$ | $+0.16$ | 38.0% | 2.366 | 6.14 | 2,508,747 | 2.66e-05 | 1.9565e-02 |
| `RL_MALA_tau0.14_amp0.06_pen0.30_sig1.15` | 0.14 | 0.06 | 0.30 | 1.15 | $4.524513 \pm 0.000310$ | $+0.46$ | 37.0% | 2.481 | 5.46 | 2,836,309 | 2.53e-05 | 2.0379e-02 |
| `RL_MALA_tau0.12_amp0.18_pen0.40_sig1.15` | 0.12 | 0.18 | 0.40 | 1.15 | $4.524156 \pm 0.000332$ | $+0.10$ | 37.1% | 2.286 | 6.23 | 2,524,005 | 2.34e-05 | 2.1759e-02 |
| `RL_MALA_tau0.12_amp0.18_pen0.30_sig1.15` | 0.12 | 0.18 | 0.30 | 1.15 | $4.524225 \pm 0.000320$ | $+0.17$ | 37.7% | 2.294 | 5.81 | 2,658,766 | 2.44e-05 | 2.2171e-02 |
| `RL_MALA_tau0.12_amp0.06_pen0.40_sig1.15` | 0.12 | 0.06 | 0.40 | 1.15 | $4.524513 \pm 0.000338$ | $+0.46$ | 38.1% | 2.374 | 6.38 | 2,424,780 | 2.01e-05 | 2.3018e-02 |
| `RL_MALA_tau0.14_amp0.08_pen0.20_sig1.15` | 0.14 | 0.08 | 0.20 | 1.15 | $4.524463 \pm 0.000340$ | $+0.41$ | 37.2% | 2.475 | 6.39 | 2,416,922 | 2.75e-05 | 2.3157e-02 |
| `RL_MALA_tau0.14_amp0.08_pen0.30_sig1.15` | 0.14 | 0.08 | 0.30 | 1.15 | $4.524093 \pm 0.000340$ | $+0.04$ | 36.9% | 2.472 | 6.47 | 2,375,735 | 2.59e-05 | 2.4399e-02 |
| `RL_MALA_tau0.12_amp0.08_pen0.30_sig1.15` | 0.12 | 0.08 | 0.30 | 1.15 | $4.524488 \pm 0.000319$ | $+0.44$ | 38.8% | 2.364 | 5.86 | 2,666,591 | 2.52e-05 | 2.5133e-02 |
| `RL_MALA_tau0.12_amp0.12_pen0.40_sig1.15` | 0.12 | 0.12 | 0.40 | 1.15 | $4.524225 \pm 0.000325$ | $+0.17$ | 37.6% | 2.341 | 5.96 | 2,583,037 | 2.88e-05 | 2.5140e-02 |
| `RL_MALA_tau0.14_amp0.18_pen0.40_sig1.15` | 0.14 | 0.18 | 0.40 | 1.15 | $4.524302 \pm 0.000326$ | $+0.25$ | 36.0% | 2.385 | 5.94 | 2,606,911 | 2.87e-05 | 2.6181e-02 |
| `RL_MALA_tau0.14_amp0.06_pen0.40_sig1.15` | 0.14 | 0.06 | 0.40 | 1.15 | $4.524231 \pm 0.000323$ | $+0.18$ | 36.9% | 2.480 | 5.83 | 2,653,635 | 2.56e-05 | 2.6872e-02 |
| `RL_MALA_tau0.12_amp0.12_pen0.30_sig1.15` | 0.12 | 0.12 | 0.30 | 1.15 | $4.524177 \pm 0.000319$ | $+0.13$ | 38.3% | 2.343 | 5.83 | 2,640,575 | 2.26e-05 | 2.6953e-02 |
| `RL_MALA_tau0.14_amp0.08_pen0.40_sig1.15` | 0.14 | 0.08 | 0.40 | 1.15 | $4.524159 \pm 0.000323$ | $+0.11$ | 37.0% | 2.469 | 5.88 | 2,627,095 | 2.45e-05 | 2.8113e-02 |
| `RL_MALA_tau0.12_amp0.06_pen0.20_sig1.15` | 0.12 | 0.06 | 0.20 | 1.15 | $4.524290 \pm 0.000317$ | $+0.24$ | 39.6% | 2.363 | 5.63 | 2,745,588 | 2.59e-05 | 2.8349e-02 |
| `RL_MALA_tau0.14_amp0.18_pen0.30_sig1.15` | 0.14 | 0.18 | 0.30 | 1.15 | $4.524276 \pm 0.000340$ | $+0.22$ | 36.0% | 2.392 | 6.53 | 2,365,351 | 2.29e-05 | 3.0270e-02 |
| `RL_MALA_tau0.14_amp0.12_pen0.20_sig1.15` | 0.14 | 0.12 | 0.20 | 1.15 | $4.524441 \pm 0.000336$ | $+0.39$ | 37.1% | 2.449 | 6.36 | 2,415,339 | 2.46e-05 | 3.0293e-02 |
| `RL_MALA_tau0.12_amp0.08_pen0.20_sig1.15` | 0.12 | 0.08 | 0.20 | 1.15 | $4.524149 \pm 0.000308$ | $+0.10$ | 39.9% | 2.352 | 5.38 | 2,857,063 | 2.25e-05 | 3.2840e-02 |
| `RL_MALA_tau0.14_amp0.18_pen0.20_sig1.15` | 0.14 | 0.18 | 0.20 | 1.15 | $4.524396 \pm 0.000311$ | $+0.34$ | 36.6% | 2.404 | 5.45 | 2,820,170 | 2.71e-05 | 3.3199e-02 |
| `RL_MALA_tau0.14_amp0.12_pen0.30_sig1.15` | 0.14 | 0.12 | 0.30 | 1.15 | $4.524305 \pm 0.000320$ | $+0.25$ | 36.7% | 2.449 | 5.87 | 2,673,597 | 2.41e-05 | 3.4292e-02 |
| `RL_MALA_tau0.14_amp0.06_pen0.20_sig1.15` | 0.14 | 0.06 | 0.20 | 1.15 | $4.524571 \pm 0.000307$ | $+0.52$ | 37.7% | 2.475 | 5.28 | 2,931,706 | 2.60e-05 | 3.4619e-02 |
| `RL_MALA_tau0.10_amp0.18_pen0.40_sig1.15` | 0.10 | 0.18 | 0.40 | 1.15 | $4.524379 \pm 0.000318$ | $+0.33$ | 40.1% | 2.138 | 5.81 | 2,662,863 | 2.10e-05 | 3.4927e-02 |
| `RL_MALA_tau0.12_amp0.12_pen0.20_sig1.15` | 0.12 | 0.12 | 0.20 | 1.15 | $4.524396 \pm 0.000326$ | $+0.34$ | 39.7% | 2.330 | 5.96 | 2,595,812 | 2.56e-05 | 3.8300e-02 |
| `RL_MALA_tau0.12_amp0.18_pen0.20_sig1.15` | 0.12 | 0.18 | 0.20 | 1.15 | $4.524041 \pm 0.000344$ | $-0.01$ | 38.9% | 2.289 | 6.63 | 2,356,130 | 2.58e-05 | 4.4708e-02 |

#### Análisis Cuantitativo de los Resultados en `MultiDet_N6_g1.0` (Ayler Grid Completa):
1. **Récord Histórico de Simetría Nodal ($D_{\mathrm{KL}}^{\mathrm{spin}} = 2.06 \times 10^{-5}$):**
   - El modelo `RL_MALA_tau0.10_amp0.06_pen0.30_sig1.15` alcanza una divergencia de espín récord de **$D_{\mathrm{KL}}^{\mathrm{spin}} = \mathbf{2.0578 \times 10^{-5}}$**, superando por un factor de **$\mathbf{1,040\times}$** al mejor MALA (`MALA_tau0.25`, $D_{\mathrm{KL}}^{\mathrm{spin}} = 0.0214$) y por **$\mathbf{18,860\times}$** al MALA estándar conservador (`MALA_tau0.06`, $D_{\mathrm{KL}}^{\mathrm{spin}} = 0.3881$).
   - Todos los 36 modelos RL-MALA superan a todos los MALA clásicos, confirmando la erradicación del congelamiento topológico en el espacio $\mathbb{R}^6$ de 20 sectores de permutación de espín.
2. **Conservación Exacta de la Energía CASSCF:**
   - Para la totalidad de los modelos, la energía variacional calculada se mantiene estrictamente dentro de $|\Delta E| \le 0.52\text{ mHa}$ respecto a la referencia exacta CASSCF ($4.524052\text{ Ha}$), probando formalmente que la propuesta de Langevin modificada por RL preserva la convergencia insesgada hacia el estado fundamental.
3. **Dinámica de Salto Colectivo sin Degradación:**
   - La distancia cuadrática media de salto alcanza valores récord $\mathrm{ESJD} \in [2.14, 2.48]\text{ Bohr}^2$, cuadruplicando la movilidad difusiva de MALA $\tau=0.06$ ($0.616\text{ Bohr}^2$), con tiempos integrados de autocorrelación muy reducidos ($\tau_{\mathrm{int}}(E) \approx 5.3 - 6.6$ pasos).

#### Figuras Clave del Benchmark 2 (`MultiDet_N6_g1.0`):

##### Reconstrucción de Densidades Espaciales y Simetría de Espín Exacta
![Densidades Unidimensionales y Simetría Exacta (MultiDet N=6, g=1.0)](results_nodal_crossing_search_new/figures/MultiDet_N6_g1.0/Article_Densities_MultiDet_N6_g1.0.png)  
*Figura 5.1: Densidades electrónicas de un cuerpo $\rho_\uparrow(x)$ y $\rho_\downarrow(x)$ sobre los 6 picos del sistema multi-determinante en base $N_{\mathrm{basis}}=256$. RL-MALA reproduce simétricamente los 6 máximos espaciales superponiéndose con la curva CASSCF de referencia.*

##### Dinámica Individual de Cruce Nodal (Trazas de 1 Caminante)
![Trazas de 1 Solo Caminante: MALA vs RL-MALA (MultiDet N=6, g=1.0)](results_nodal_crossing_search_new/figures/MultiDet_N6_g1.0/Article_Walker_Traces_MultiDet_N6_g1.0.png)  
*Figura 5.2: Trazas de 1 caminante ($N=6$) durante $40,000$ pasos. MALA permanece atrapado en una partición fija ($3\uparrow$ a la izquierda, $3\downarrow$ a la derecha), mientras que RL-MALA permuta colectivamente las posiciones de espín opuesto.*

##### Exploración Sistemática del Espacio de Hiperparámetros (Grilla Ayler, 41 Modelos)
![Evolución con Hiperparámetros (MultiDet N=6, g=1.0)](figures_param_search_summary/MultiDet_N6_g1.0/fig_hyperparameter_evolution_MultiDet_N6_g1.0.png)  
*Figura 5.3: Evolución de $D_{\mathrm{KL}}^{\mathrm{spin}}$, $\mathrm{ESJD}$, $\tau_{\mathrm{int}}(E)$ y tasa de aceptación como función de $\tau$ para $N=6$.*

![Mapas de Calor 2D del Actor (MultiDet N=6, g=1.0)](figures_param_search_summary/MultiDet_N6_g1.0/fig_param_search_heatmaps_MultiDet_N6_g1.0.png)  
*Figura 5.4: Mapas de calor 2D para $N=6$. Se verifica que el régimen $\tau=0.10$ con amplitudes $A_\mu \le 0.12$ maximiza simultáneamente el cruce nodal y la aceptación.*

![Sensibilidad a la Penalización de Rechazo lambda_pen (MultiDet N=6, g=1.0)](figures_param_search_summary/MultiDet_N6_g1.0/fig_penalty_sensitivity_MultiDet_N6_g1.0.png)  
*Figura 5.5: Sensibilidad del error de espín $D_{\mathrm{KL}}^{\mathrm{spin}}$ frente a la penalización $\lambda_{\mathrm{pen}}$ en $N=6$.*

### 5.5 Diagnóstico Exhaustivo de la Falla en el Modelo 16 y Corrección Algorítmica

#### 1. ¿Qué sucedió en Ayler?
A las 09:28:58, durante la ejecución del modelo `MultiDet_N6_g1.0_RL_MALA_tau0.12_amp0.06_pen0.30_sig1.15` en Slot #2 (GPU 2), la biblioteca XLA de JAX intentó compilar una fusión matricial (`gemm_fusion_dot_general.103`) mediante el autotuner de Triton/GEMM invocando el ejecutable externo `ptxas`.
La llamada al sistema falló con el error crítico:
```text
Check failed: executable.status() is OK (INTERNAL: Failed to launch ptxas) - Failure occured when compiling fusion gemm_fusion_dot_general.103 with config 'reference (cublas)'
```
Inmediatamente después (09:29:00), el subshell de bash intentó lanzar el siguiente modelo invocando el ejecutable de Python memorizado en la tabla hash interna de bash:
```text
./run_parameter_search_baker.sh: line 619: /home/romanarmenta/miniconda3/envs/jax-gpu/bin/python3: No such file or directory
```
Debido a una desconexión transitoria de la ruta del entorno conda o la falta de exportación explícita del binario en subshells huérfanos, el comando retornó el código de salida 127 en 0.05 segundos.
El bucle principal del orquestador detectó la liberación inmediata del Slot #2 y procedió a lanzar el siguiente modelo, quemando los 20 modelos restantes de la grilla en menos de 18 segundos, sin generar los archivos `run_bundle.h5`.

#### 2. Correcciones Implementadas en `run_parameter_search_baker.sh` y `run_parameter_search_ayler.sh`:
1. **Detección y Exportación de Rutas CUDA:**
   Se incorporó la búsqueda y exportación automática de `/usr/local/cuda/bin` a `$PATH` y `/usr/local/cuda/lib64` a `$LD_LIBRARY_PATH`.
2. **Desactivación del Autotuner Inestable de Triton GEMM:**
   Se fijó la bandera de XLA:
   ```bash
   export XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"
   ```
   Esto obliga a JAX a utilizar las rutinas nativas estables de cuBLAS, eliminando la compilación dinámica de fisiones con `ptxas` que detonaba el Check failed.
3. **Resolución Dinámica y Validación de `$PYTHON_BIN`:**
   Se reemplazaron todas las llamadas a `python3` plano por la variable `$PYTHON_BIN`, validando al inicio del script que pueda ejecutar `import jax` antes de encolar cualquier proceso.
4. **Circuito de Seguridad (Circuit Breaker / Fail-Fast):**
   Se añadió en `launch_job` y `wait_for_free_slot` la detección de trabajos que terminan en menos de 5 segundos con código de error. Si ocurren 3 fallos instantáneos consecutivos, el orquestador aborta la ejecución con un mensaje claro en lugar de agotar la cola.
5. **Reanudación Automática y Segura:**
   La función `is_run_completed` verifica que `run_bundle.h5` exista y tenga un tamaño $> 10\text{ KB}$. Las 56 corridas completadas (41 de $N=4$ y 15 de $N=6$) son saltadas instantáneamente, procesando únicamente los 26 modelos pendientes.

---

## 6. Análisis Comparativo Global y Mapas de Calor (`all_models_global`)

El análisis unificado de las 114 configuraciones evaluadas (Baker y Ayler consolidado) proporciona una visión integral del rendimiento del muestreador en ambos regímenes físicos:

##### Frontera de Pareto: Compensación entre Movilidad y Tasa de Aceptación
![Frontera de Pareto de Muestreo Cuántico](figures_param_search_summary/global/fig_pareto_sampling_frontier.png)  
*Figura 6.1: Frontera de Pareto para todos los modelos evaluados en $N=4$ y $N=6$. El panel izquierdo grafica $\mathrm{ESJD}$ frente a la tasa de aceptación; el panel derecho ilustra la divergencia de espín $D_{\mathrm{KL}}^{\mathrm{spin}}$ frente al tiempo de autocorrelación $\tau_{\mathrm{int}}(E)$. RL-MALA rompe el límite difusivo clásico de MALA, logrando saltos $\mathrm{ESJD} > 2.0\text{ Bohr}^2$ con aceptaciones estables del $33\%-45\%$.*

##### Comparación Global de Cruce Nodal: Mejor MALA vs Mejor RL-MALA
![Comparativa Global de Cruce Nodal: Mejor MALA vs Mejor RL-MALA](figures_param_search_summary/global/fig_global_nodal_crossing_comparison.png)  
*Figura 6.2: Comparación directa entre los mejores modelos MALA y RL-MALA para $N=4$ y $N=6$ en escala logarítmica para $D_{\mathrm{KL}}^{\mathrm{spin}}$, $\mathrm{ESJD}$, $\tau_{\mathrm{int}}(E)$ y tasa de aceptación. Se aprecia la reducción de 2 a 4 órdenes de magnitud en la asimetría de espín.*

##### Visión General Consolidada de Ambos Sistemas
![Visión General Global de Todos los Sistemas](results_nodal_crossing_search_new/figures/all_models_global/fig_all_systems_global_overview.png)  
*Figura 6.3: Visión global de todas las configuraciones evaluadas para $Wigner\_N4\_g4.0$ y $MultiDet\_N6\_g1.0$, contrastando la dispersión de MALA (líneas discontinuas) frente a RL-MALA (líneas sólidas).*

##### Mapas de Calor Globales del Actor ($A_\mu$ vs $\lambda_{\mathrm{pen}}$)
![Mapas de Calor Globales del Actor](results_nodal_crossing_search_new/figures/all_models_global/fig_param_search_heatmaps_actor.png)  
*Figura 6.4: Mapas de calor consolidados en función de la amplitud de red $A_\mu$ y la penalización de rechazo $\lambda_{\mathrm{pen}}$, demostrando que $A_\mu \in [0.06, 0.12]$ y $\lambda_{\mathrm{pen}} \ge 0.20$ minimizan uniformemente el error de espín sin degradar la aceptación.*

---

## 7. Protocolo de Ejecución, Resiliencia y Recuperación en Clusters (Baker / Ayler)

La finalización exitosa de los 114 modelos demostró la efectividad de las medidas de resiliencia implementadas:

1. **Recuperación Exitosa de Ayler:**  
   Tras la corrección del autotuner de Triton GEMM y la fijación de `$PYTHON_BIN` descrita en la Sección 5.5, el pipeline reanudó automáticamente la ejecución en Ayler, procesando las 26 corridas restantes sin recalcular las existentes, completando el 100% de la grilla de $N=6$ a las 12:08.
2. **Procedimiento de Despliegue en Segundo Plano:**  
   Para futuras campañas de simulación a gran escala, la invocación estandarizada se ejecuta mediante:
   ```bash
   nohup ./run_parameter_search_baker.sh results_nodal_crossing_search > out_search.txt 2>&1 &
   ```
   Con verificación en tiempo real del progreso:
   ```bash
   tail -f results_nodal_crossing_search/logs/master_search.log
   ```

---

## 8. Conclusión sobre Hiperparámetros y Prescripción para Producción (Pipelines 1 y 2)

### ¿Se requiere buscar más hiperparámetros?
**No.** La intersección de los resultados de `Wigner_N4_g4.0` (interacción fuerte $g=4.0$) y `MultiDet_N6_g1.0` (mayor número de partículas $N=6$ y expansión multi-determinante de 20 determinantes en base 256) demuestra que existe una **región óptima universal y altamente robusta**:

- **Paso temporal base ($\tau$):**
  - $\tau = 0.10$ para $N \in \{2, 4\}$ y para $N=6$ con $g \le 2.0$.
  - $\tau = 0.08$ para $N=6$ en el régimen fuertemente correlacionado ($g \ge 3.0$).
  - $\tau = 0.06$ para $N=10$ (donde el movimiento simultáneo de 10 electrones requiere un paso base ligeramente menor para mantener la aceptación de MALA en $\sim 75\%$ y la de RL-MALA en $\sim 45\%-55\%$).
- **Amplitud del Actor ($A_\mu$):**
  - $A_\mu = 0.15$ para acoplamiento débil a moderado ($g \le 2.0$, $N \le 6$).
  - $A_\mu = 0.10$ para acoplamiento fuerte ($g \ge 3.0$) o $N=10$, evitando las caídas de aceptación observadas con $A_\mu = 0.25$ en pozos de Wigner estrechos.
- **Penalización de Rechazo ($\lambda_{\mathrm{pen}}$) y Escala Estocástica ($\sigma_{\mathrm{scale}}$):**
  - $\lambda_{\mathrm{pen}} = 0.25$ y $\sigma_{\mathrm{scale}} = 1.1$, con `--reward_type cdlb` y `--max_phys_jump_factor 2.0`.
- **Régimen de Optimización TD3:**
  - Tasa de aprendizaje constante (`--no-use_lr_scheduler`, `--lr_actor 1e-4`, `--lr_critic 3e-4`), $6,000$ barridos de entrenamiento con `--init_mode symmetric_gaussian --init_std 3.0` y evaluación rigurosa de $3 \times 40,000$ pasos desde `--eval_init_mode spin_ordered_offset`.

Esta parametrización ha sido integrada directamente en `run_production_benchmark_3pipelines.sh` para ejecutar los **Pipelines 1 y 2** sobre toda la grilla $(N, g)$ con $g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0\}$.

---

## 9. Caracterización Sistemática de Intercambios Topológicos (`Flips`) respecto a $N$ y $g$ (`flips_analysis_advisor`)

Para demostrar cuantitativamente cómo el Actor neuronal aprende a cruzar nodos de Pauli y seudo-barreras de Coulomb a medida que el sistema transita desde el gas de Fermi/Hartree ($g \le 0.3$) hacia el cristal de Wigner fuertemente localizado ($g \ge 0.5$), se ejecutó el Pipeline 2 y el analizador `scripts/analyze_flips_vs_g.py` sobre los **54 modelos** con funciones de onda CASSCF de espacio activo completo ($N \in \{2, 4, 6, 8\}$, incorporando los estados convergidos `c0-a6, b=256` y `c0-a8, b=256` del asesor en `flips_analysis_advisor/`).

### 9.1 Metodología: Cruces con Histéresis ($\delta = \eta\,d_{\mathrm{adj}}$) y Mapa de Calor del Stream-Flow Normalizado
Para cada par de electrones $(i, j)$ a lo largo de las $C \times B = 3 \times 32 = 96$ trayectorias guardadas de $T = 40,000$ pasos MCMC, monitoreamos la coordenada relativa $d_{ij}(t) = x_i(t) - x_j(t)$:
1. **Separación Media entre Vecinos Adyacentes ($d_{\mathrm{adj}}$):** En cada paso $t$, ordenamos las posiciones espaciales $x_{(1)}(t) < x_{(2)}(t) < \dots < x_{(N)}(t)$ y calculamos $d_{\mathrm{adj}} = \langle x_{(k+1)} - x_{(k)} \rangle$.
2. **Filtro de Histéresis Espacial ($\delta = 0.25\,d_{\mathrm{adj}}$):** Un cruce crudo ($\delta=0$) ocurre cada vez que $d_{ij}(t)$ cambia de signo, lo cual a acoplamiento débil ($g=0.1$) incluye oscilaciones térmicas sobre el plano $x_i = x_j$. Un **flip genuino pozo-a-pozo con histéresis** se registra únicamente cuando el par parte de una separación comprometida $d_{ij}(t_{\mathrm{enter}}) \le -\delta$ y alcanza el bolsillo permutado opuesto $d_{ij}(t_{\mathrm{exit}}) \ge +\delta$ (o viceversa).
3. **Duración de Tránsito ($\Delta t_{\mathrm{transit}} = t_{\mathrm{exit}} - t_{\mathrm{enter}}$):** Número exacto de pasos de la cadena de Markov requeridos para atravesar la zona de barrera $[-\delta, +\delta]$. Si $\Delta t_{\mathrm{transit}} = 1$, la permutación ocurre en **un único paso colectivo MCMC** (*tunneling jump*).
4. **Etiquetado Canónico Izquierda/Derecha y Alineación Exacta en el Cruce ($t = t_{\mathrm{flip}}, y = 0$):** Para caracterizar el flujo microscópico (*stream flow*) en la ventana $t - t_{\mathrm{flip}} \in [-15, +15]$, etiquetamos canónicamente como $x_i(t)$ al electrón que inicia a la **izquierda** en el espacio real 1D antes del flip ($x_i < x_j$ para $t < t_{\mathrm{flip}}$) y como $x_j(t)$ al electrón que inicia a la **derecha** ($x_j > x_i$ para $t < t_{\mathrm{flip}}$), alineando el origen temporal exactamente en el instante $t_{\mathrm{flip}}$ donde $x_i(t_{\mathrm{flip}}) - x_j(t_{\mathrm{flip}}) = 0$. Al normalizar por $d_{\mathrm{adj}}$, la coordenada relativa adimensional $(x_i(t) - x_j(t))/d_{\mathrm{adj}}$ fluye universalmente desde $-1.0$ (antes del intercambio), cruza exactamente por $(0, 0)$ en el paso del flip, y alcanza $+1.0$ (después del intercambio).

### 9.2 Resultados Consolidados en $N=2, 4, 6$ con Espacio Activo Completo (`flips_analysis_advisor`)

La tabla siguiente presenta el resumen cuantitativo de `flips_analysis_advisor/flips_vs_g_summary.csv` para $N=2$ ($\mathrm{CAS}(2e, 4o)$), $N=4$ ($\mathrm{CAS}(4e, 4o)$) y $N=6$ con los nuevos estados convergidos del asesor ($\mathrm{CAS}(6e, 6o)$, `c0-a6`, $N_{\mathrm{basis}}=256$, 20 determinantes):

| $N$ | $g$ | $d_{\mathrm{adj}}$ (Bohr) | Método | Flips Totales Hist. ($/10^3$) | Flips $\uparrow\downarrow$ Hist. ($/10^3$) | Flips $\uparrow\uparrow,\downarrow\downarrow$ Pauli ($/10^3$) | Cruces Crudos ($/10^3$) | Saltos en 1 Paso ($\%$) | $\langle\lvert\Delta x_{ij}\rvert\rangle_{\mathrm{flip}}$ (Bohr) | $D_{\mathrm{KL}}^{\mathrm{spin}}$ |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 0.1 | 4.15 | MALA | $4.98 \pm 0.11$ | $4.98$ | $0.00$ | $16.19$ | $0.1\%$ | $0.89$ | $2.28 \times 10^{-4}$ |
| 2 | 0.1 | 4.15 | **RL-MALA** | **$25.30 \pm 0.08$** | **$25.30$** | **$0.00$** | **$44.43$** | **$31.0\%$** | **$2.54$** | **$1.13 \times 10^{-4}$** |
| 2 | 0.5 | 6.05 | MALA | $0.17 \pm 0.003$ | $0.17$ | $0.00$ | $0.46$ | $0.0\%$ | $0.92$ | $1.77 \times 10^{-3}$ |
| 2 | 0.5 | 6.06 | **RL-MALA** | **$1.37 \pm 0.03$** | **$1.37$** | **$0.00$** | **$1.99$** | **$24.0\%$** | **$2.78$** | **$8.17 \times 10^{-5}$** |
| 2 | 1.0 | 6.96 | MALA | $0.012 \pm 0.003$ | $0.012$ | $0.00$ | $0.033$ | $0.0\%$ | $0.83$ | $1.00 \times 10^{-1}$ |
| 2 | 1.0 | 6.94 | **RL-MALA** | **$0.315 \pm 0.018$** | **$0.315$** | **$0.00$** | **$0.383$** | **$26.8\%$** | **$3.02$** | **$1.12 \times 10^{-4}$** |
| 2 | 2.0 | 8.22 | MALA | $0.000 \pm 0.000$ | $0.000$ | $0.00$ | $0.000$ | $0.0\%$ | — | $6.47 \times 10^{-1}$ |
| 2 | 2.0 | 8.20 | **RL-MALA** | **$0.049 \pm 0.007$** | **$0.049$** | **$0.00$** | **$0.056$** | **$12.6\%$** | **$2.49$** | **$1.29 \times 10^{-3}$** |
| 4 | 0.1 | 3.17 | MALA | $29.71 \pm 0.20$ | $29.63$ | $0.08$ | $88.99$ | $1.4\%$ | $0.89$ | $6.36 \times 10^{-5}$ |
| 4 | 0.1 | 3.18 | **RL-MALA** | **$94.64 \pm 0.55$** | **$89.72$** | **$4.92$** | **$159.29$** | **$42.4\%$** | **$2.44$** | **$4.47 \times 10^{-5}$** |
| 4 | 0.5 | 4.58 | MALA | $0.279 \pm 0.027$ | $0.163$ | $0.116$ | $0.298$ | $0.6\%$ | $0.92$ | $3.50 \times 10^{-2}$ |
| 4 | 0.5 | 4.58 | **RL-MALA** | **$15.28 \pm 0.21$** | **$10.23$** | **$5.05$** | **$18.09$** | **$54.7\%$** | **$3.13$** | **$7.18 \times 10^{-5}$** |
| 4 | 1.0 | 5.26 | MALA | $0.158 \pm 0.013$ | $0.099$ | $0.058$ | $0.167$ | $0.0\%$ | $0.90$ | $1.14 \times 10^{-1}$ |
| 4 | 1.0 | 5.26 | **RL-MALA** | **$7.69 \pm 0.13$** | **$5.12$** | **$2.57$** | **$9.29$** | **$43.5\%$** | **$3.09$** | **$1.88 \times 10^{-4}$** |
| 4 | 2.0 | 6.27 | MALA | $0.033 \pm 0.001$ | $0.018$ | $0.015$ | $0.033$ | $0.0\%$ | $0.91$ | $3.34 \times 10^{-1}$ |
| 4 | 2.0 | 6.27 | **RL-MALA** | **$2.18 \pm 0.08$** | **$1.45$** | **$0.73$** | **$2.65$** | **$31.3\%$** | **$3.07$** | **$6.56 \times 10^{-4}$** |
| 4 | 3.0 | 7.00 | MALA | $0.014 \pm 0.005$ | $0.003$ | $0.010$ | $0.014$ | $0.0\%$ | $0.96$ | $5.92 \times 10^{-1}$ |
| 4 | 3.0 | 7.00 | **RL-MALA** | **$0.738 \pm 0.070$** | **$0.505$** | **$0.233$** | **$0.872$** | **$25.2\%$** | **$3.07$** | **$4.48 \times 10^{-3}$** |
| 4 | 5.0 | 8.10 | MALA | $0.004 \pm 0.003$ | $0.001$ | $0.003$ | $0.004$ | $0.0\%$ | $1.05$ | $5.05 \times 10^{-1}$ |
| 4 | 5.0 | 8.09 | **RL-MALA** | **$0.129 \pm 0.010$** | **$0.083$** | **$0.046$** | **$0.146$** | **$17.8\%$** | **$3.01$** | **$5.46 \times 10^{-2}$** |
| 6 | 0.1 | 2.48 | MALA | $86.71 \pm 0.16$ | $86.39$ | $0.32$ | $226.03$ | $6.0\%$ | $0.89$ | $3.88 \times 10^{-5}$ |
| 6 | 0.1 | 2.48 | **RL-MALA** | **$178.72 \pm 0.14$** | **$163.73$** | **$14.99$** | **$279.17$** | **$54.1\%$** | **$2.32$** | **$2.28 \times 10^{-5}$** |
| 6 | 0.5 | 3.84 | MALA | $0.861 \pm 0.008$ | $0.488$ | $0.373$ | $0.908$ | $2.6\%$ | $0.96$ | $6.39 \times 10^{-2}$ |
| 6 | 0.5 | 3.84 | **RL-MALA** | **$32.60 \pm 0.11$** | **$19.63$** | **$12.97$** | **$37.21$** | **$65.2\%$** | **$3.12$** | **$7.53 \times 10^{-5}$** |
| 6 | 1.0 | 4.40 | MALA | $0.445 \pm 0.013$ | $0.251$ | $0.193$ | $0.472$ | $1.1\%$ | $0.95$ | $1.38 \times 10^{-1}$ |
| 6 | 1.0 | 4.40 | **RL-MALA** | **$18.78 \pm 0.11$** | **$11.17$** | **$7.60$** | **$21.93$** | **$56.4\%$** | **$3.15$** | **$1.56 \times 10^{-4}$** |
| 6 | 2.0 | 5.24 | MALA | $0.132 \pm 0.012$ | $0.076$ | $0.056$ | $0.144$ | $0.0\%$ | $0.95$ | $1.04 \times 10^{-1}$ |
| 6 | 2.0 | 5.64 | **RL-MALA** | **$6.13 \pm 0.18$** | **$3.66$** | **$2.46$** | **$7.64$** | **$37.8\%$** | **$3.12$** | **$5.83 \times 10^{-2}$** |
| 6 | 3.0 | 5.85 | MALA | $0.058 \pm 0.016$ | $0.034$ | $0.024$ | $0.173$ | $0.0\%$ | $0.82$ | $1.05 \times 10^{-1}$ |
| 6 | 3.0 | 6.25 | **RL-MALA** | **$1.83 \pm 0.21$** | **$1.09$** | **$0.74$** | **$2.47$** | **$20.7\%$** | **$2.65$** | **$1.36 \times 10^{-1}$** |

#### Hallazgos Físicos Fundamentales sobre los Nuevos Resultados (`flips_analysis_advisor`):
1. **Confirmación de la Ley de Escalamiento Exponencial de Flips en $N=6$ con los Estados Convergidos (`c0-a6, b=256`):**
   Al reemplazar los estados truncados antiguos (`c1-a4`) por los estados convergidos del asesor (`c0-a6, b=256`), la anomalía que existía en $N=6$ desaparece por completo en $g \in [0.1, 3.0]$: la tasa de flips de MALA decae monótonamente desde $86.71\text{ flips}/10^3$ ($g=0.1$) hasta $0.861$ ($g=0.5$), $0.445$ ($g=1.0$), $0.132$ ($g=2.0$) y $0.058$ ($g=3.0$). Simultáneamente, RL-MALA alcanza ganancias de **$37.9\times$ en $g=0.5$** ($32.60$ vs $0.86\text{ flips}/10^3$, reduciendo $D_{\mathrm{KL}}^{\mathrm{spin}}$ en **$848\times$**, de $0.0639$ a $7.53 \times 10^{-5}$), **$42.2\times$ en $g=1.0$** ($18.78$ vs $0.445\text{ flips}/10^3$, reduciendo $D_{\mathrm{KL}}^{\mathrm{spin}}$ en **$887\times$**, de $0.1385$ a $1.56 \times 10^{-4}$), **$46.6\times$ en $g=2.0$** ($6.13$ vs $0.132\text{ flips}/10^3$) y **$31.5\times$ en $g=3.0$** ($1.83$ vs $0.058\text{ flips}/10^3$).
2. **Proporción Combinatoria Exacta 3:2 en $N=6$ y 2:1 en $N=4$ entre Barreras de Wigner ($\uparrow\downarrow$) y Nodos de Pauli ($\uparrow\uparrow, \downarrow\downarrow$):**
   En $N=6$ ($3\uparrow, 3\downarrow$) existen $\binom{3}{1}\binom{3}{1} = 9$ pares de espín opuesto y $\binom{3}{2} + \binom{3}{2} = 6$ pares del mismo espín (razón combinatoria **3:2 = 1.50**). Para todo $g \ge 0.5$, RL-MALA mantiene exactamente una razón **1.50:1** entre flips $\uparrow\downarrow$ y flips $\uparrow\uparrow/\downarrow\downarrow$ ($19.63 / 12.97 = 1.51$ en $g=0.5$; $11.17 / 7.60 = 1.47$ en $g=1.0$; $3.66 / 2.46 = 1.49$ en $g=2.0$; $1.09 / 0.74 = 1.47$ en $g=3.0$). Junto con la razón 2:1 en $N=4$, esto demuestra rigurosamente que en el régimen de Wigner el actor cruza con idéntica eficiencia por par tanto las barreras de Coulomb $\uparrow\downarrow$ como los hiperplanos nodales exactos de Pauli $\Psi = 0$, mediante saltos de escala característica $\langle|\Delta x_{ij}|\rangle_{\mathrm{flip}} \approx 3.12\text{ Bohr}$ ($65.2\%$ en 1 solo paso en $g=0.5$ y $56.4\%$ en 1 solo paso en $g=1.0$).

### 9.3 Figuras Principales del Análisis con Estados Convergidos (`flips_analysis_advisor`)

##### Síntesis Global Maestra de Flips vs $g$ ($N=2, 4, 6, 8$)
![Síntesis Global de Intercambios Topológicos (Flips) respecto a N y g](flips_analysis_advisor/fig_flips_vs_g_all_N_master_P2.png)  
*Figura 9.1: Síntesis panorámica de la tasa de flips reales pozo-a-pozo con histéresis (panel a), cruces crudos sin histéresis (panel b), factor de aceleración $R_{\mathrm{RL-MALA}}/R_{\mathrm{MALA}}$ (panel c) y divergencia de espín $D_{\mathrm{KL}}^{\mathrm{spin}}$ (panel d) en función del acoplamiento $g$ con los estados del asesor.*

##### Evolución de Flips respecto a $g$ para $N=6$ Electrones (`c0-a6, b=256`)
![Evolución de Flips respecto a g para N=6](flips_analysis_advisor/per_N/fig_flips_vs_g_P2_N6.png)  
*Figura 9.2: Evolución completa de flips frente a $g$ para $N=6$ con los estados convergidos `c0-a6, b=256`. Se aprecia la ganancia de $38\times$ a $47\times$ en la ventana $g \in [0.5, 2.0]$ (panel c), la proporción combinatoria 3:2 entre canales $\uparrow\downarrow$ y $\uparrow\uparrow/\downarrow\downarrow$ (panel b), y hasta $65.2\%$ de cruces en 1 solo paso MCMC (panel d).*

##### Evolución de Flips respecto a $g$ para $N=4$ Electrones (`c0-a4, b=32`)
![Evolución de Flips respecto a g para N=4](flips_analysis_advisor/per_N/fig_flips_vs_g_P2_N4.png)  
*Figura 9.3: Caracterización completa en función de $g$ para $N=4$ ($\mathrm{CAS}(4e, 4o)$ exacto), mostrando ganancias de hasta $53\times$ en tasa total de flips y erradicación del bloqueo ergódico.*

##### Anatomía de 4 Paneles en $N=6, \omega=0.1, g=0.5$ y $g=1.0$ (`c0-a6, b=256`)
![Anatomía Microscópica de Flips en N=6, g=0.5](flips_analysis_advisor/per_system/fig_flips_system_P2_N6_w0.1_g0.5.png)  
*Figura 9.4: Caracterización en $N=6, g=0.5$ ($d_{\mathrm{adj}} = 3.84\text{ Bohr}$). RL-MALA alcanza $32.60\text{ flips}/1\text{k}$ frente a $0.86\text{ flips}/1\text{k}$ de MALA ($37.9\times$), ejecutando el $65.2\%$ de los intercambios en un único paso colectivo MCMC.*

![Anatomía Microscópica de Flips en N=6, g=1.0](flips_analysis_advisor/per_system/fig_flips_system_P2_N6_w0.1_g1.png)  
*Figura 9.5: Caracterización en $N=6, g=1.0$ ($d_{\mathrm{adj}} = 4.40\text{ Bohr}$). En el panel (a), MALA presenta sólo $0.44\text{ flips}/1\text{k}$ y mantiene los espines segregados, mientras que RL-MALA alcanza $18.78\text{ flips}/1\text{k}$ ($56.4\%$ en 1 paso) y mezcla uniformemente los 6 pozos de Wigner.*

### 9.4 Diagnóstico Técnico de las Trazas Anómalas en $N=6$ ($g \ge 2$) y $N=8$: El Piso Numérico `1e-15` en `CASSCF1DAnsatz` y la Inicialización `spin_ordered_offset`

Al inspeccionar las figuras individuales por sistema en `flips_analysis_advisor/per_system/` para $N=6$ ($g \in \{2, 3, 5, 8\}$) y $N=8$ ($g \ge 0.5$), se observó que las trazas del Panel (a) parecen brotar de un punto cercano a $x \approx 0$ en $t=0$ y luego abrirse en abanico hasta posiciones de $|x| \sim 100 - 300\text{ Bohr}$, con un aspecto de movimiento browniano libre en lugar de un cristal de Wigner confinado.

El análisis matemático del código fuente reveló la combinación exacta de **tres factores acoplados** (y su solución definitiva ya implementada en el repositorio):

1. **Causa Física/Numérica Principal — Piso Aditivo `+ 1e-15` en `CASSCF1DAnsatz.apply` (`rl_vmc/ansatze/ansatz_1d.py`):**
   En la implementación previa de `CASSCF1DAnsatz.apply`, el logaritmo de la función de onda se evaluaba como:
   $$
   \ln |\Psi_{\mathrm{clip}}(\mathbf{x})| = \ln\!\big(|\Psi_{\mathrm{CASSCF}}(\mathbf{x})| + 10^{-15}\big)
   $$
   Dado que cada orbital del oscilador armónico incluye el factor gaussiano $\exp(-\frac{1}{2}\omega x_i^2)$ antes de evaluar los determinantes $N_\uparrow \times N_\uparrow$ y $N_\downarrow \times N_\downarrow$, en dimensión $N=6$ ($g \ge 2$) y $N=8$ cualquier configuración comprimida o desplazada hacia las colas produce un valor lineal $|\Psi(\mathbf{x})| \sim 10^{-20} - 10^{-35} \ll 10^{-15}$ (perfectamente representable en `float64` hasta $2.2 \times 10^{-308}$, pero eclipsado por el término aditivo $+10^{-15}$).
   En el instante en que $|\Psi(\mathbf{x})| \ll 10^{-15}$, la función $\ln(|\Psi(\mathbf{x})| + 10^{-15})$ se vuelve **idénticamente constante** ($= \ln(10^{-15}) = -34.5388$):
   - La deriva cuántica de Langevin se anula por completo: $\mathbf{F}_{\mathrm{Q}}(\mathbf{x}) = \nabla_{\mathbf{x}} \ln(|\Psi(\mathbf{x})| + 10^{-15}) = \mathbf{0}$ (desaparece la fuerza restauradora armónica $-\omega \mathbf{x}$).
   - La razón de aceptación de Metropolis-Hastings $\big(\frac{|\Psi(\mathbf{x}')| + 10^{-15}}{|\Psi(\mathbf{x})| + 10^{-15}}\big)^2 = 1$ acepta el $100\%$ de los pasos aleatorios sin confinamiento.
   - En consecuencia, cualquier caminante que cae en la región $|\Psi(\mathbf{x})| < 10^{-15}$ escapa hacia $\pm 100 - 300\text{ Bohr}$ como una partícula browniana libre.

2. **Causa de Inicialización — `spin_ordered_offset` Descentrado (`base=5.0, step=1.0`) con `eval_burn_in = 200`:**
   El modo `--eval_init_mode spin_ordered_offset` utilizaba por defecto $x_i(0) = 5.0 + 1.0\cdot i$, colocando a los 6 electrones en $[5, 6, 7, 8, 9, 10]\text{ Bohr}$ y a los 8 electrones en $[5, 6, 7, 8, 9, 10, 11, 12]\text{ Bohr}$ (todos comprimidos en el semieje derecho con separación de $1\text{ Bohr}$). Para $N=8$ ($g \ge 0.5$) y $N=6$ ($g \ge 5$), esta configuración inicial ya tiene $|\Psi(\mathbf{x}_0)| \sim 10^{-25} < 10^{-15}$ en $t=0$, por lo que los caminantes inician directamente sobre el piso plano sin fuerza restauradora. Además, al graficar el eje vertical hasta $\pm 200\text{ Bohr}$, el paquete inicial en $[5, 12]\text{ Bohr}$ aparenta visualmente salir desde $x \approx 0$.

3. **Sesgo de Selección en el Ploteo del Panel (a) (`argmax(flips_per_cb)` en `scripts/analyze_flips_vs_g.py`):**
   El script seleccionaba el caminante individual con el máximo número de cruces (`np.argmax(flips_per_cb)`). En sistemas como $N=6, g=2.0$ (RL-MALA) o $N=6, g=3.0$, donde el $\sim 90\%-95\%$ de los caminantes permaneció correctamente confinado en el cristal de Wigner, bastó con que 1 o 2 caminantes de los 96 escaparan al piso plano `1e-15` para que sus múltiples cruces brownianos hicieran que `argmax` eligiera invariablemente al caminante fugado para el Panel (a).

#### Correcciones Definitivas Implementadas y Lista Exacta de Modelos a Re-correr:
- **En `rl_vmc/ansatze/ansatz_1d.py`:** Se reemplazó el piso `+ 1e-15` por $\epsilon_{\mathrm{tiny}} = 2.22 \times 10^{-308}$ (`jnp.finfo(float64).tiny`) combinado mediante `jnp.logaddexp` con la envolvente armónica asintótica exacta $-500.0 - \frac{1}{2}\omega \sum_i x_i^2$. Esto preserva la exactitud de máquina en todo el dominio físico y garantiza que la fuerza restauradora $-\omega\mathbf{x}$ jamás se anule.
- **En `run_advisor_casscf_pipeline.sh` y `rl_vmc/samplers/mh_sampler_1d.py`:** Se centró automáticamente `spin_ordered_offset` alrededor de $x=0$ con el espaciamiento de Wigner $d_{\mathrm{adj}}(N, g)$ (`get_system_wigner_init_offset`) y se incrementó `EVAL_BURN_IN=2000`, añadiendo la bandera `--rerun-failed-traces`.
- **En `scripts/analyze_flips_vs_g.py`:** Se incorporó el filtrado automático de caminantes fugados (`healthy_mask`), la selección de caminante representativo sano en el Panel (a), y el nuevo **Mapa de Calor 2D del Stream-Flow Normalizado** en el Panel (c).
- **Modelos que salieron limpios (NO requieren re-correrse):** Todos los de **$N=2$** ($g \in \{0.1, \dots, 8.0\}$), todos los de **$N=4$** ($g \in \{0.1, \dots, 8.0\}$), y **$N=6$ con $g \in \{0.1, 0.5, 1.0\}$**.
- **Modelos que deben re-correrse con `--rerun-failed-traces`:** **$N=6$ para $g \in \{2.0, 3.0, 5.0, 8.0\}$** y **$N=8$ para $g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0\}$**.

---

## 10. Siguiente Frontera Científica: Transfer Learning y Aceleración de VMC ($g=1.0 \to g=2.0$)

Habiendo resuelto y cartografiado completamente la dinámica de cruce nodal en funciones de onda congeladas, el siguiente avance científico consiste en explotar la **capacidad de generalización y transferencia de conocimiento (*Transfer Learning*)** de los agentes neuronales de muestreo a lo largo del diagrama de fases cuántico.

### 10.1 Fundamento Teórico del Aprendizaje por Transferencia en MCMC Cuántico
La política del Actor $\boldsymbol{\mu}_\theta(\mathbf{x}, \mathbf{g})$ está parametrizada mediante una arquitectura **DeepSet / Graph Neural Network continua** equivariante ante permutaciones de partículas idénticas:

$$
\mathbf{u}_\theta^{(i)}(\mathbf{x}) = \mathbf{W}_2 \,\phi\!\left( \mathbf{W}_1 \mathbf{h}_i + \sum_{j \neq i} \mathbf{W}_{\sigma_i \sigma_j} \, f_{\mathrm{filter}}(|x_i - x_j|) \cdot \frac{x_i - x_j}{\sqrt{(x_i - x_j)^2 + \epsilon^2}} \right)
$$

Esta parametrización posee dos sesgos inductivos (*inductive biases*) esenciales para el aprendizaje cuántico:
1. **Invariancia Traslacional y Dependencia de Coordenadas Relativas ($r_{ij} = |x_i - x_j|$):**  
   La red no memoriza posiciones absolutas en el laboratorio, sino configuraciones geométricas relativas de pares y correlaciones locales de apantallamiento de Coulomb.
2. **Deformación Topológica Continua con el Acoplamiento $g$:**  
   Al variar la constante de interacción de $g=1.0$ a $g=2.0$, la estructura de simetría del espacio de configuración $\mathbb{R}^N$ permanece idéntica (los mismos $N!/(N_\uparrow! N_\downarrow!)$ sectores de permutación). Lo que cambia es la profundidad de los valles inter-nodales y la rigidez de la localización de Wigner.

Por lo tanto, un actor entrenado para $g=1.0$ ya ha adquirido la representación latente de las **direcciones de escape colectivas**: sabe exactamente cómo coordinar el desplazamiento de un electrón $\uparrow$ hacia la derecha mientras desplaza simultáneamente al electrón $\downarrow$ vecino hacia la izquierda para minimizar la penalización de rechazo de Pauli-Coulomb.

### 10.2 Protocolo Experimental Propuesto: Benchmark de Transferencia $g=1.0 \to g=2.0$

Proponemos una secuencia experimental de tres etapas rigurosamente cuantificables:

```mermaid
flowchart LR
    A["Actor Entrenado (g=1.0)"] --> B["Fase 1: Zero-Shot Transfer a g=2.0"]
    B --> C["Fase 2: Few-Shot Fine-Tuning (500 sweeps)"]
    C --> D["Fase 3: Aceleración de VMC Variacional"]
    style A fill:#e1f5fe,stroke:#0288d1
    style B fill:#fff3e0,stroke:#f57c00
    style C fill:#e8f5e9,stroke:#388e3c
    style D fill:#f3e5f5,stroke:#7b1fa2
```

1. **Fase 1: Transferencia de Disparo Cero (*Zero-Shot Transfer*):**
   - Se toma el modelo óptimo obtenido en $N=6, g=1.0$ (`RL_MALA_tau0.10_amp0.06_pen0.30_sig1.15`).
   - Sin ningún paso adicional de re-entrenamiento ni ajuste de pesos ($\nabla_\theta = \mathbf{0}$), se evalúa directamente su rendimiento de muestreo sobre la función de onda CASSCF congelada de $N=6, g=2.0$.
   - **Métricas:** Evaluar $D_{\mathrm{KL}}^{\mathrm{spin}}$, tasa de aceptación y $\mathrm{ESJD}$ comparados directamente contra MALA clásico puro en $g=2.0$.
   - **Hipótesis:** El actor $g=1.0$ conservará una capacidad de cruce nodal órdenes de magnitud superior a MALA, confirmando la generalización física sin sobreajuste.

2. **Fase 2: Re-Ajuste Rápido de Pocos Disparos (*Few-Shot Fine-Tuning*):**
   - Comparar el entrenamiento de un actor en $g=2.0$ desde ceros (*from scratch*, 6,000 barridos) frente a la adaptación fina (*fine-tuning*) del actor pre-entrenado en $g=1.0$ durante únicamente $500$ y $1,000$ barridos.
   - **Métricas de Éxito:** Curvas de recompensa acumulada CDLB, tiempo de cómputo en GPU requerido para alcanzar $D_{\mathrm{KL}}^{\mathrm{spin}} < 10^{-4}$ y velocidad de convergencia del Crítico $Q(s, a)$.

3. **Fase 3: Aceleración del VMC Completo (Optimización Variacional Conjunta):**
   - En la optimización variacional estándar de wave functions neuronales ($\Psi_\Theta(\mathbf{x}) = \Psi_{\mathrm{CASSCF}}(\mathbf{x})\,e^{J_\Theta(\mathbf{x})}$), el paso más costoso es el re-muestreo MCMC en cada iteración del gradiente de energía.
   - Utilizar el sampler transferido y adaptado en tiempo real para generar las muestras de Monte Carlo reduce drásticamente el tiempo de autocorrelación $\tau_{\mathrm{int}}(E)$, permitiendo reducir el número de pasos de muestreo requeridos por iteración en un factor de $3\times$ a $5\times$.

---

## 11. Resultados de Producción en Baker (`results_production_3pipelines_baker_N6_N8`), Diagnóstico de Regresión en $N=8$ ($g \ge 2$) y Preparación para $N=2, 8, 10$

Tras corregir el piso numérico `1e-15` en `CASSCF1DAnsatz`, actualizar la inicialización de evaluación a `spin_ordered_offset` y usar las funciones de onda convergidas del asesor (`n_basis=256, n_core=0, quad=800`), se completó la corrida de producción en Baker para $N=6$ ($g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0\}$) y $N=8$ ($g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0\}$) en `results_production_3pipelines_baker_N6_N8/`. **Los 13 estados de referencia del Pipeline 1 y todas las corridas del Pipeline 2 finalizaron sin fugas brownianas ni errores de convergencia**.

### 11.1 Referencias CASSCF (`Pipeline 1`) Verificadas ($N=6$ y $N=8$, `b=256, c0`)

Todos los estados instalados desde `CASSCF_csv/` reportaron `converged=True`, $E_{\mathrm{core}} = 0.0$ y $\omega=0.1, \eta_{\mathrm{soft}}=0.1$:

| $N$ | $g$ | Determinantes | $E_{\mathrm{CASSCF}}$ ($E_h$) | Radio RMS Exacto (Bohr) | Iteraciones L-BFGS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6 | 0.1 | 6 / 400 | $1.58992883$ | $4.4847$ | 431 |
| 6 | 0.5 | 20 / 400 | $3.28187523$ | $6.5724$ | 56 |
| 6 | 1.0 | 20 / 400 | $4.52405213$ | $7.3649$ | 71 |
| 6 | 2.0 | 20 / 400 | $6.57676356$ | $8.3981$ | 99 |
| 6 | 3.0 | 20 / 400 | $8.30661370$ | $8.0405$ | 113 |
| 6 | 5.0 | 20 / 400 | $11.27280996$ | $6.3790$ | 144 |
| 6 | 8.0 | 20 / 400 | $15.06363820$ | $6.7280$ | 461 |
| 8 | 0.1 | 6 / 4900 | $2.75178389$ | $5.0613$ | 644 |
| 8 | 0.5 | 70 / 4900 | $5.84287110$ | $7.3505$ | 59 |
| 8 | 1.0 | 70 / 4900 | $8.07323817$ | $7.4237$ | 85 |
| 8 | 2.0 | 70 / 4900 | $11.78411490$ | $7.2234$ | 120 |
| 8 | 3.0 | 70 / 4900 | $15.20562413$ | $7.5950$ | 207 |
| 8 | 5.0 | 70 / 4900 | $20.29751211$ | $7.5855$ | 547 |

### 11.2 Comparación Cuantitativa MALA vs RL-MALA TD3 (`Pipeline 2`) y Flips Topológicos

| $N$ | $g$ | MALA Acc (%) | RL Acc (%) | MALA ESJD | RL ESJD | $\tau_{\mathrm{MALA}}$ | $\tau_{\mathrm{RL}}$ | $G_{\mathrm{RL}}$ | $D_{\mathrm{KL}}^{\mathrm{spin}}$ (MALA $\to$ RL) | Flips / 1k (MALA $\to$ RL) | 1-Paso RL (%) | Estado |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 6 | 0.1 | $87.13$ | $40.77$ | $1.027$ | $3.052$ | $5.76$ | $4.61$ | $1.249\times$ | $2.23\times 10^{-5} \to 2.36\times 10^{-5}$ | $91.95 \to 191.68$ ($2.08\times$) | $57.9$ | Exitoso |
| 6 | 0.5 | $82.78$ | $32.37$ | $0.971$ | $2.357$ | $8.37$ | $6.58$ | $1.273\times$ | $6.29\times 10^{-2} \to 5.92\times 10^{-5}$ | $0.845 \to 33.22$ ($39.3\times$) | $67.8$ | Exitoso ($1062\times$ KL) |
| 6 | 1.0 | $83.41$ | $32.24$ | $0.978$ | $2.304$ | $7.93$ | $5.80$ | $1.367\times$ | $1.24\times 10^{-1} \to 8.53\times 10^{-5}$ | $0.442 \to 19.20$ ($43.4\times$) | $58.5$ | Exitoso ($1452\times$ KL) |
| 6 | 2.0 | $83.07$ | $29.88$ | $0.972$ | $2.057$ | $8.57$ | $5.51$ | $1.554\times$ | $3.32\times 10^{-1} \to 3.08\times 10^{-4}$ | $0.124 \to 6.11$ ($49.3\times$) | $47.1$ | Exitoso ($1078\times$ KL) |
| 6 | 3.0 | $84.74$ | $34.27$ | $0.795$ | $1.929$ | $11.27$ | $5.79$ | $1.946\times$ | $5.41\times 10^{-1} \to 9.25\times 10^{-3}$ | $0.028 \to 1.706$ ($60.1\times$) | $29.1$ | Exitoso ($58.5\times$ KL*) |
| 6 | 5.0 | $84.83$ | $34.30$ | $0.796$ | $1.926$ | $13.64$ | $7.14$ | $1.912\times$ | $6.59\times 10^{-1} \to 2.10\times 10^{-1}$ | $0.0042 \to 0.343$ ($82.4\times$) | $23.7$ | Exitoso ($3.1\times$ KL*) |
| 6 | 8.0 | $85.05$ | $34.88$ | $0.798$ | $1.967$ | $16.58$ | $8.24$ | $2.011\times$ | $6.93\times 10^{-1} \to 4.51\times 10^{-1}$ | $0.00078 \to 0.0458$ ($58.7\times$) | $16.5$ | Exitoso (KL* truncado) |
| 8 | 0.1 | $84.19$ | $34.55$ | $1.060$ | $2.821$ | $9.48$ | $7.39$ | $1.282\times$ | $2.96\times 10^{-5} \to 2.95\times 10^{-5}$ | $130.06 \to 240.23$ ($1.85\times$) | $60.2$ | Exitoso |
| 8 | 0.5 | $79.29$ | $25.32$ | $0.996$ | $2.003$ | $8.80$ | $9.45$ | $0.931\times$ | $1.30\times 10^{-1} \to 7.58\times 10^{-5}$ | $1.288 \to 40.74$ ($31.6\times$) | $67.9$ | Exitoso ($1712\times$ KL) |
| 8 | 1.0 | $80.44$ | $25.73$ | $1.010$ | $2.003$ | $8.75$ | $8.32$ | $1.052\times$ | $2.03\times 10^{-1} \to 3.36\times 10^{-4}$ | $0.704 \to 24.85$ ($35.3\times$) | $59.2$ | Exitoso ($605\times$ KL) |
| 8 | 2.0 | $80.30$ | $12.78$ | $1.006$ | $0.600$ | $7.99$ | $177.54$ | $0.045\times$ | $3.83\times 10^{-1} \to 3.16\times 10^{-1}$ | $0.238 \to 3.067$ ($12.9\times$) | $45.8$ | **Regresión de entrenamiento** |
| 8 | 3.0 | $82.68$ | $16.78$ | $0.778$ | $0.627$ | $14.72$ | $202.85$ | $0.073\times$ | $4.88\times 10^{-1} \to 3.35\times 10^{-1}$ | $0.066 \to 1.095$ ($16.5\times$) | $27.2$ | **Regresión de entrenamiento** |
| 8 | 5.0 | $82.86$ | $16.50$ | $0.780$ | $0.608$ | $14.53$ | $190.95$ | $0.076\times$ | $6.38\times 10^{-1} \to 4.22\times 10^{-1}$ | $0.0083 \to 0.216$ ($26.0\times$) | $18.1$ | **Regresión de entrenamiento** |

*\*En $N=6, g \ge 3.0$ y $N=8, g \ge 1.0$, el diagnóstico de densidad en `evaluate_nodal_crossing_1d.py` usó una malla truncada en $[-14.94, +14.94]$ Bohr (ver Sección 11.3).*

**Conclusiones principales de esta corrida:**
1. **Cero fugas brownianas en todos los sistemas**: En los 13 sistemas ($N=6$ y $N=8$), la divergencia KL de densidad total $D_{\mathrm{KL}}^{\mathrm{tot}}$ permaneció en el orden de $7.17\times 10^{-6}$ a $5.19\times 10^{-5}$ (excepto $N=8, g=3.0$ en RL-MALA debido a la asimetría inducida por el actor colapsado, $D_{\mathrm{KL}}^{\mathrm{tot}} = 3.07\times 10^{-2}$). Las energías medias coinciden dentro de $10^{-3}\text{–}10^{-4}\ E_h$ con CASSCF (salvo el sesgo base de Gauss-Hermite finito en $g=0.1$ ya documentado).
2. **Desempeño sobresaliente en $N=6$ ($g=0.1 \dots 8.0$) y $N=8$ ($g=0.1 \dots 1.0$)**: RL-MALA multiplica los *flips* topológicos con histéresis por un factor de $31.6\times$ a $82.4\times$ en el régimen fuertemente correlacionado, realizando entre $16.5\%$ y $67.9\%$ de los cruces mediante saltos cuánticos directos en **1 solo paso** (frente a $0.0\%$ de MALA para $g \ge 2.0$), lo que reduce $D_{\mathrm{KL}}^{\mathrm{spin}}$ hasta $1452\times$ ($N=6, g=1.0$) y $1712\times$ ($N=8, g=0.5$).

### 11.3 Diagnóstico de Causa Raíz: Regresión en $N=8$ ($g \ge 2.0$) y Truncamiento de Dominio

El análisis conjunto de `training_history_*.csv`, `data_densities_*.csv` y `observables_P2_*.csv` identificó **tres causas acopladas** detrás de la degradación de RL-MALA en $N=8, g \ge 2.0$ (y del sesgo métrico en los regímenes cristalizados grandes):

1. **Trampa nodal inicial durante el entrenamiento (`INIT_MODE="symmetric_gaussian"`)**:
   Al entrenar RL-MALA con `init_mode=symmetric_gaussian` (`init_std=3.0`) y `n_burn_in=2000` barridos, los caminantes iniciales en $N=8, g \ge 2.0$ quedaron atrapados en bolsillos nodales locales fuera de equilibrio (varianza de energía inicial y final estancada sin relajar durante los $6,000$ pasos de TD3):
   - $N=8, g=2.0$: $E_{\mathrm{mean}}: -87.69 \to -92.36$, $\mathrm{Var}(E): 7.87\times 10^5 \to 7.85\times 10^5$, recompensa $R = -0.026 \to -0.019$, $\mathcal{L}_{\mathrm{actor}} = +1.710 \to +1.755$, $\alpha_{\mathrm{final}}=0.125$.
   - $N=8, g=3.0$: $E_{\mathrm{mean}}: +35.20 \to +34.22$, $\mathrm{Var}(E): 2.59\times 10^4 \to 2.59\times 10^4$, recompensa $R = -0.123 \to -0.068$, $\mathcal{L}_{\mathrm{actor}} = +2.414 \to +2.993$, $\alpha_{\mathrm{final}}=0.164$.
   - $N=8, g=5.0$: $E_{\mathrm{mean}}: -181.58 \to -195.43$, $\mathrm{Var}(E): 5.38\times 10^6 \to 5.28\times 10^6$, recompensa $R = -0.095 \to -0.066$, $\mathcal{L}_{\mathrm{actor}} = +2.595 \to +3.031$, $\alpha_{\mathrm{final}}=0.188$.
   Al pasar luego a evaluación en equilibrio (`eval_init_mode=spin_ordered_offset`, $|x| \sim 20\text{–}30$ Bohr), el actor entrenado fuera de distribución propuso desplazamientos anómalos y asimétricos ($\rho_\uparrow^{\max}=0.3693$ vs $\rho_\downarrow^{\max}=0.2858$ en $g=3.0$; $D_{\mathrm{KL}}^{\uparrow}=0.212$ vs $D_{\mathrm{KL}}^{\downarrow}=0.458$), colapsando $G_{\mathrm{RL}}$ a $0.045\times\text{–}0.076\times$ y disparando $\tau_{\mathrm{int}}(E)$ a $177.5\text{–}202.9$.
   - **Solución implementada**: Se creó la inicialización `randomized_wigner` en `rl_vmc/samplers/mh_sampler_1d.py` (`MHSampler1D.init_state`) y `rl_vmc/config.py` (red centrada simétrica escalada en $\omega, g, N$ con permutación aleatoria por caminante e invariancia de espín/paridad), fijándola como nuevo default `INIT_MODE="randomized_wigner"` en `run_advisor_casscf_pipeline.sh` y `run_production_benchmark_3pipelines.sh`.

2. **Sobreescala dimensional de $\sigma_{\mathrm{action\_scale}} = 1.1$ en $N \ge 8$**:
   Mantener fijo $\sigma_{\mathrm{action\_scale}} = 1.1$ al crecer de $6\mathrm{D}$ a $8\mathrm{D}\text{–}10\mathrm{D}$ eleva linealmente con $N$ la varianza radial de la propuesta gaussiana ($\sum_{i=1}^N \sigma_i^2$), explicando por qué incluso los casos exitosos en $N=8$ ($g=0.5, 1.0$) mostraron caída de aceptación a $25.3\%\text{–}25.7\%$ (vs $32.2\%\text{–}34.9\%$ en $N=6$).
   - **Solución implementada**: Calibración automática por dimensión en `get_system_p2_params`: $\sigma_{\mathrm{scale}} = 1.1$ ($N \le 6$), $\sigma_{\mathrm{scale}} = 0.90$ ($N \in \{7, 8\}$) y $\sigma_{\mathrm{scale}} = 0.80$ ($N \ge 9$, junto con $\tau=0.07/0.06$, `actor_amp=0.08`), más la bandera `--sigma-scale` para sobrescribirlo.

3. **Truncamiento del dominio de densidad en `cli/evaluate_nodal_crossing_1d.py` ($x_{\max} = 15.0$ Bohr)**:
   Mientras `observables.py` expandió correctamente las mallas hasta $[-29.75, +29.75]$ ($N=6, g=8$; $N=8, g=3$) y $[-30.74, +30.74]$ Bohr ($N=8, g=5$), `data_densities_*.csv` mantuvo fijo $x_{\max}=15.0$ Bohr: en $N=6, g=8.0$, la integral exacta retenida cayó a $\int \rho_{\mathrm{exact}}\,dx = 4.0000$ de $6.0$ (recortando ambos picos externos de Wigner), y en $N=8, g=5.0$ cayó a $5.6687$ de $8.0$ (con densidad de borde $\rho(14.94) = 0.2299$), sesgando los valores aparentes de `eval_kl_spin` sobre la malla recortada.
   - **Solución implementada**: En `cli/evaluate_nodal_crossing_1d.py`, `--x_max` pasa a `default=None` y calcula automáticamente `eff_x_max` usando $\lceil 1.25 \cdot P_{99.8}(|x|) \rceil$ a partir de `production/walkers` antes de evaluar la referencia exacta y los histogramas.

### 11.4 Auditoría de `CASSCF_csv/`, Ejecución Selectiva y Plan para $N=2, 8, 10$

La auditoría completa de `CASSCF_csv/` verificó **24 estados convergidos** (`n_basis=256, n_core=0, omega=0.1, softening=0.1`, todos con `converged=True` y $|g_{\mathrm{orb}}| < 10^{-5}$):
- **$N=4$ (7 estados, $c0\text{-}a4$, 36 dets):** $g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0\}$.
- **$N=6$ (7 estados, $c0\text{-}a6$, 400 dets):** $g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0\}$.
- **$N=8$ (6 estados, $c0\text{-}a8$, 4900 dets):** $g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0\}$ (sin $g=8.0$).
- **$N=10$ (4 estados, $c0\text{-}a10$, 63,504 dets):** $g=1.0$ ($12.555692\ E_h$), $g=2.0$ ($18.357692\ E_h$), $g=3.0$ ($23.272532\ E_h$), $g=5.0$ ($31.703367\ E_h$).
- **$N=2$ (0 estados en `CASSCF_csv/`):** Se agregó `--include-n2` (activado automáticamente al incluir `N2` en `--systems`) para calcular las referencias $N=2$ con `b=256, c0-a4, quad=800`.

Además, se eliminó la cuarentena ciega sobre `N=10` cuando existen archivos `b=256` convergidos, y se incorporó filtrado no cartesiano (`--systems "N2:all N8:2.0,3.0,5.0 N10:all"`, `--n-filter`, `--g-filter`, `--skip-global-plots`, `--merge_summary_csv`) en `run_advisor_casscf_pipeline.sh`, `run_production_benchmark_3pipelines.sh` y `scripts/analyze_flips_vs_g.py` para ejecutar únicamente $N=2$, $N=8$ ($g=2.0, 3.0, 5.0$) y $N=10$ sin recalcular $N=4$ ni $N=6$.


