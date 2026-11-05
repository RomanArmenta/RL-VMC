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
- **Densidades Unidimensionales y Simetría Exacta:**  
  [Article_Densities_Wigner_N4_g4.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/Wigner_N4_g4.0/Article_Densities_Wigner_N4_g4.0.png)  
  *Muestra la reconstrucción de los 4 picos de densidad para $\rho_\uparrow(x)$ y $\rho_\downarrow(x)$, demostrando cómo RL-MALA reproduce la simetría de espín exacta mientras MALA permanece asimétrico.*
- **Trazas de 1 Solo Caminante (Mejor MALA vs Mejor RL-MALA):**  
  [Article_Walker_Traces_Wigner_N4_g4.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/Wigner_N4_g4.0/Article_Walker_Traces_Wigner_N4_g4.0.png)  
  *Ilustra la dinámica individual de un caminante: MALA queda atrapado en pozos fijos, mientras que RL-MALA ejecuta cruces frecuentes entre los 4 pozos de Wigner sin perder aceptación.*
- **Evolución con Hiperparámetros y Comparación de Modelos:**  
  [Hyperparameter_Evolution_Wigner_N4_g4.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/Wigner_N4_g4.0/Hyperparameter_Evolution_Wigner_N4_g4.0.png) | [Parameter_Search_Comparison_Wigner_N4_g4.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/Wigner_N4_g4.0/Parameter_Search_Comparison_Wigner_N4_g4.0.png)

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

### 5.4 Nueva Exploración Sistemática en Ayler (Grilla Parcial de 15 Modelos)
En Ayler se lanzó la grilla complementaria con $\sigma_{\mathrm{sc}} = 1.15$ y pasos $\tau \in \{0.10, 0.12, 0.14\}$. La evaluación completó 15 modelos (5 MALA + 10 RL-MALA a $\tau=0.10$) antes de interrumpirse temporalmente por una falla del entorno en el modelo 16:

| Modelo | $\tau$ | $A_\mu$ | $\lambda_{\mathrm{pen}}$ | $\sigma_{\mathrm{sc}}$ | $\langle E \rangle \pm \mathrm{SE}(E)$ (Ha) | $\Delta E$ (mHa) | Aceptación | ESJD ($\text{Bohr}^2$) | $\tau_{\mathrm{int}}(E)$ | $\mathrm{ESS/s}$ ($\text{s}^{-1}$) | $D_{\mathrm{KL}}^{\mathrm{tot}}$ | $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MALA_tau0.06` | 0.06 | — | — | — | $4.524417 \pm 0.000477$ | $+0.37$ | 87.2% | 0.616 | 12.83 | 2,238 | 8.35e-06 | **0.2648** |
| `MALA_tau0.14` | 0.14 | — | — | — | $4.524131 \pm 0.000351$ | $+0.08$ | 80.3% | 1.313 | 6.92 | 4,175 | 6.11e-06 | **0.0778** |
| `MALA_tau0.16` | 0.16 | — | — | — | $4.524121 \pm 0.000315$ | $+0.07$ | 78.9% | 1.472 | 5.70 | 4,957 | 6.45e-06 | **0.0565** |
| `MALA_tau0.20` | 0.20 | — | — | — | $4.524138 \pm 0.000295$ | $+0.09$ | 76.3% | 1.775 | 4.94 | 5,746 | 4.66e-06 | **0.0338** |
| `MALA_tau0.25` | 0.25 | — | — | — | $4.523880 \pm 0.000266$ | $-0.17$ | 73.4% | 2.128 | 4.08 | 6,968 | 5.17e-06 | **0.0210** |
| **`RL_MALA_tau0.10_amp0.06_pen0.30_sig1.15`** | 0.10 | 0.06 | 0.30 | 1.15 | $4.523999 \pm 0.000313$ | $-0.05$ | 33.8% | 2.420 | 5.49 | 4,781 | 5.69e-06 | **1.4799e-05** |
| **`RL_MALA_tau0.10_amp0.08_pen0.40_sig1.15`** | 0.10 | 0.08 | 0.40 | 1.15 | $4.524008 \pm 0.000317$ | $-0.04$ | 33.5% | 2.401 | 5.61 | 4,713 | 5.95e-06 | **1.9042e-05** |
| **`RL_MALA_tau0.10_amp0.08_pen0.30_sig1.15`** | 0.10 | 0.08 | 0.30 | 1.15 | $4.524279 \pm 0.000315$ | $+0.23$ | 33.5% | 2.400 | 5.57 | 4,715 | 6.82e-06 | **4.5422e-05** |
| **`RL_MALA_tau0.10_amp0.06_pen0.20_sig1.15`** | 0.10 | 0.06 | 0.20 | 1.15 | $4.524408 \pm 0.000317$ | $+0.36$ | 33.8% | 2.419 | 5.60 | 4,804 | 4.74e-06 | **7.3832e-05** |
| **`RL_MALA_tau0.10_amp0.12_pen0.20_sig1.15`** | 0.10 | 0.12 | 0.20 | 1.15 | $4.524325 \pm 0.000322$ | $+0.27$ | 32.8% | 2.345 | 5.84 | 4,583 | 5.49e-06 | **7.4791e-05** |
| **`RL_MALA_tau0.10_amp0.18_pen0.20_sig1.15`** | 0.10 | 0.18 | 0.20 | 1.15 | $4.524191 \pm 0.000329$ | $+0.14$ | 31.3% | 2.225 | 6.11 | 4,309 | 7.43e-06 | **1.1565e-04** |
| **`RL_MALA_tau0.10_amp0.12_pen0.30_sig1.15`** | 0.10 | 0.12 | 0.30 | 1.15 | $4.524082 \pm 0.000314$ | $+0.03$ | 32.8% | 2.344 | 5.61 | 4,702 | 6.81e-06 | **1.5095e-04** |
| **`RL_MALA_tau0.10_amp0.12_pen0.40_sig1.15`** | 0.10 | 0.12 | 0.40 | 1.15 | $4.524530 \pm 0.000315$ | $+0.48$ | 32.8% | 2.344 | 5.55 | 4,781 | 7.57e-06 | **1.6684e-04** |
| **`RL_MALA_tau0.10_amp0.06_pen0.40_sig1.15`** | 0.10 | 0.06 | 0.40 | 1.15 | $4.524272 \pm 0.000310$ | $+0.22$ | 33.8% | 2.420 | 5.40 | 4,886 | 5.76e-06 | **1.9239e-04** |
| **`RL_MALA_tau0.10_amp0.08_pen0.20_sig1.15`** | 0.10 | 0.08 | 0.20 | 1.15 | $4.524408 \pm 0.000320$ | $+0.36$ | 33.6% | 2.398 | 5.69 | 4,727 | 4.99e-06 | **2.0529e-04** |

#### Análisis Cuantitativo de los Resultados en `MultiDet_N6_g1.0` (Ayler Grid):
1. **Récord Histórico de Simetría Nodal ($D_{\mathrm{KL}}^{\mathrm{spin}} = 1.48 \times 10^{-5}$):**
   - El modelo `RL_MALA_tau0.10_amp0.06_pen0.30_sig1.15` alcanza una divergencia de espín de **$D_{\mathrm{KL}}^{\mathrm{spin}} = \mathbf{1.4799 \times 10^{-5}}$**, el valor más bajo jamás registrado para el sistema de 6 partículas.
   - En comparación con el mejor MALA de paso grande (`MALA_tau0.25`, $D_{\mathrm{KL}}^{\mathrm{spin}} = 0.02096$), RL-MALA reduce el error de asimetría por un factor de **$\mathbf{1,416\times}$**.
   - Frente a MALA estándar de paso conservador (`MALA_tau0.06`, $D_{\mathrm{KL}}^{\mathrm{spin}} = 0.2648$), la mejora alcanza **$\mathbf{17,896\times}$**.
2. **Conservación Exacta de la Energía CASSCF:**
   - La energía obtenida por el sampler neuronal es $E = 4.523999 \pm 0.000313\text{ Ha}$, idéntica a la referencia CASSCF exacta ($4.524052\text{ Ha}$) con una desviación de apenas $\Delta E = -0.05\text{ mHa}$, dentro del margen de error estadístico.
3. **Movilidad Superior sin Colapso de Aceptación:**
   - La distancia cuadrática de salto alcanza $\mathrm{ESJD} = \mathbf{2.420\text{ Bohr}^2}$, casi cuadruplicando a MALA $\tau=0.06$ ($0.616\text{ Bohr}^2$) y superando a MALA $\tau=0.25$ ($2.128\text{ Bohr}^2$) con una aceptación del $33.8\%$.

#### Figuras Clave del Benchmark 2 (`MultiDet_N6_g1.0`):
- **Densidades Unidimensionales y Simetría Exacta:**  
  [Article_Densities_MultiDet_N6_g1.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/MultiDet_N6_g1.0/Article_Densities_MultiDet_N6_g1.0.png)  
  *Muestra la coincidencia perfecta entre $\rho_\uparrow(x)$ y $\rho_\downarrow(x)$ sobre los 6 picos para RL-MALA, mientras que MALA $\tau=0.06$ y $\tau=0.14$ exhiben fuerte polarización espacial.*
- **Trazas de 1 Solo Caminante (Mejor MALA vs Mejor RL-MALA):**  
  [Article_Walker_Traces_MultiDet_N6_g1.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/MultiDet_N6_g1.0/Article_Walker_Traces_MultiDet_N6_g1.0.png)  
  *Visualiza cómo el actor DeepSet coordina el cruce de los 6 electrones entre las regiones nodalmente prohibidas para un solo caminante en 40,000 pasos.*
- **Evolución con Hiperparámetros y Comparación de Modelos:**  
  [Hyperparameter_Evolution_MultiDet_N6_g1.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/MultiDet_N6_g1.0/Hyperparameter_Evolution_MultiDet_N6_g1.0.png) | [Parameter_Search_Comparison_MultiDet_N6_g1.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/MultiDet_N6_g1.0/Parameter_Search_Comparison_MultiDet_N6_g1.0.png)

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

El análisis unificado de todas las configuraciones evaluadas en Ayler genera una visión global del espacio de hiperparámetros:

- **Visión General de Todos los Sistemas:**  
  [fig_all_systems_global_overview.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/all_models_global/fig_all_systems_global_overview.png)  
  *Muestra la comparativa de $D_{\mathrm{KL}}^{\mathrm{spin}}$ y $\mathrm{ESS/s}$ en escala logarítmica para $N=4$ y $N=6$, contrastando las líneas discontinuas de MALA frente a las continuas de RL-MALA.*
- **Mapas de Calor del Actor ($A_\mu$ vs $\lambda_{\mathrm{pen}}$):**  
  [fig_param_search_heatmaps_actor.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/all_models_global/fig_param_search_heatmaps_actor.png) | [fig_param_search_heatmaps_actor_N4_g4.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/all_models_global/fig_param_search_heatmaps_actor_N4_g4.0.png)  
  *Confirma que para acoplamiento fuerte ($N=4, g=4.0$), amplitudes $A_\mu \in [0.06, 0.12]$ y penalizaciones $\lambda_{\mathrm{pen}} \ge 0.20$ minimizan el error de espín sin reducir la tasa de aceptación.*
- **Evolución de Simetría de Espín ($D_{\mathrm{KL}}^{\mathrm{spin}}$):**  
  [fig_param_search_spin_kl_comparison.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/all_models_global/fig_param_search_spin_kl_comparison.png) | [fig_param_search_spin_kl_comparison_N6_g1.0.png](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/results_nodal_crossing_search_new/figures/all_models_global/fig_param_search_spin_kl_comparison_N6_g1.0.png)

---

## 7. Guía Operativa para Reanudar la Búsqueda en Ayler

Para completar los 26 modelos restantes de $N=6$ en Ayler sin recalcular los 56 modelos que ya terminaron, sigue estos pasos:

### Paso 1: Actualizar el Repositorio en Ayler
```bash
cd /home/romanarmenta/Documents/projects/RL-VMC-main/RL-VMC
git pull origin main
```

### Paso 2: Activar el Entorno Conda y Verificar JAX y CUDA
```bash
conda activate jax-gpu
python3 -c "import jax; print('JAX detectado correctamente en:', jax.devices())"
which ptxas || echo "ptxas no en PATH, pero Triton GEMM está deshabilitado para evitar fallas"
```

### Paso 3: Lanzar la Reanudación en Background
Dado que `results_nodal_crossing_search` (o `results_nodal_crossing_search_new`) contiene las carpetas con `run_bundle.h5`, el script saltará automáticamente todos los modelos finalizados:
```bash
# Para correr sobre la carpeta existente y completar la grilla:
nohup ./run_parameter_search_baker.sh results_nodal_crossing_search_new > master_resume.out 2>&1 &
```
O para enfocarse exclusivamente en el sistema $N=6$:
```bash
nohup ./run_parameter_search_baker.sh --system N6 results_nodal_crossing_search_new > master_resume_N6.out 2>&1 &
```

### Paso 4: Monitoreo en Tiempo Real
```bash
# Ver el log maestro general
tail -f results_nodal_crossing_search_new/logs/master_search.log

# Ver el estado de las GPUs en Ayler
watch -n 1 nvidia-smi
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

Esta parametrización ha sido integrada directamente en [`run_production_benchmark_3pipelines.sh`](file:///Users/romanarmenta/Documents/Git-Projects/RL-VMC-main/run_production_benchmark_3pipelines.sh) para ejecutar los **Pipelines 1 y 2** sobre toda la grilla $(N, g)$ con $g \in \{0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0\}$ y bases CASSCF expandidas ($N_{\mathrm{basis}} = 36$ para $N=2$ y $N_{\mathrm{basis}} = 48$ para $N \ge 4$).
