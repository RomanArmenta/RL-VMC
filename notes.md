Aquí tienes las notas teóricas detalladas y rigurosas, estructuradas para que puedas utilizarlas directamente en tu artículo o presentación. Las ecuaciones reflejan la arquitectura exacta y los trucos numéricos implementados en tu código.

---

### 1. Forma General del Ansatz y Optimización de Energía (VMC)

El algoritmo busca encontrar el estado base resolviendo la ecuación de Schrödinger mediante Variational Monte Carlo (VMC). La función de onda de prueba $\Psi_\theta(\mathbf{R})$ se parametriza mediante un Ansatz de redes neuronales profundas (DeepQMCAnsatz).

* **Arquitectura:** Por defecto, el sistema inicializa una arquitectura basada en `PauliNet` acoplada con transformaciones de *backflow* (ej. multiplicativas) y una suma de $N_{det}$ determinantes de Slater (por defecto 10) para garantizar la antisimetría fermiónica.


* **Precisión Numérica:** Para evitar truncamientos fatales cerca de las singularidades de Coulomb, toda la evaluación del Ansatz y los cálculos en JAX se fuerzan estrictamente a doble precisión (`JAX_ENABLE_X64 = "True"`).



La optimización de los parámetros del Ansatz $\theta$ se realiza minimizando la energía local $E_L(\mathbf{R})$:


$$E_L(\mathbf{R}) = \frac{\hat{H} \Psi_\theta(\mathbf{R})}{\Psi_\theta(\mathbf{R})}$$

**Truco Numérico (Recorte de Energía):** Para estabilizar los gradientes, la energía local se recorta (*clipping*) numéricamente:

1. Se aplica un límite estricto absoluto entre `energy_clip_min` y `energy_clip_max` (ej. $[-50, 0]$).


2. Se calcula la desviación absoluta mediana (MAD) y se recorta estadísticamente dentro de una ventana definida por la mediana $\pm \sigma \times \text{MAD}$ (donde $\sigma$ es `sigma_energy_clip = 5.0`).



La función de pérdida para el Ansatz utiliza la aproximación estándar de la derivada logarítmica:


$$\mathcal{L}_{ansatz}(\theta) = 2 \mathbb{E}_{\mathbf{R} \sim \vert{}\Psi\vert{}^2} \left[ (E_L(\mathbf{R}) - \bar{E}) \nabla_\theta \ln \vert{}\Psi_\theta(\mathbf{R})\vert{} \right]$$


Donde el gradiente $\nabla_\theta \ln \vert{}\Psi_\theta\vert{}$ es acotado entre `min_clip_grads_ansatz` y `max_clip_grads_ansatz` (usualmente $[-20, 20]$).

---

### 2. El Algoritmo Híbrido RL-MALA

El muestreo de posiciones electrónicas $\mathbf{R}$ abandona la heurística estática del Metropolis-Adjusted Langevin Algorithm (MALA) y utiliza un Actor neuronal que propone saltos asimétricos e informados.

La propuesta del caminante para el paso de tiempo $\tau$ se define como:


$$\mathbf{R}' = \mathbf{R} + \tau \nabla_{\mathbf{R}} \ln \vert{}\Psi_\theta(\mathbf{R})\vert{} + \mathbf{A}_\phi(\mathbf{R}) + \sqrt{2\tau} \mathbf{L}_\phi(\mathbf{R}) \mathbf{\eta}$$

* $\tau \nabla_{\mathbf{R}} \ln \vert{}\Psi\vert{}$ es la deriva cuántica base limitando los gradientes a $\pm 20$.


* $\mathbf{A}_\phi(\mathbf{R})$ es la deriva (drift) direccional aprendida por el Actor.


* $\mathbf{L}_\phi(\mathbf{R})$ es la matriz de covarianza (factor de Cholesky) aprendida por el Actor, multiplicada por el ruido gaussiano isotrópico $\mathbf{\eta} \sim \mathcal{N}(0, \mathbf{I})$.



**Asimetría del Kernel y Aceptación:**
Dado que la covarianza es una matriz tensorial, la probabilidad de transición $q(\mathbf{R}' \vert{} \mathbf{R})$ se evalúa resolviendo el sistema lineal $\mathbf{y} = \mathbf{L}_\phi^{-1} (\mathbf{R}' - \mathbf{\mu}_{prop})$:


$$\ln q(\mathbf{R}' \vert{} \mathbf{R}) = -0.5 \Vert{}\mathbf{y}\Vert{}^2 - \sum_{i} \ln(L_{ii})$$


El log-ratio de aceptación de Metropolis-Hastings se define como:


$$\ln \rho = 2 \ln \vert{}\Psi(\mathbf{R}')\vert{} - 2 \ln \vert{}\Psi(\mathbf{R})\vert{} + \ln q(\mathbf{R} \vert{} \mathbf{R}') - \ln q(\mathbf{R}' \vert{} \mathbf{R})$$


*Truco de Estabilidad:* Este `log_ratio` es purgado de valores nulos (NaN) e infinitos (reemplazados por $\pm 20$) y finalmente acotado estrictamente en el intervalo $[-20, 20]$ antes de la aceptación. La aceptación se determina estocásticamente si $\ln(u) < \text{safe\_log\_ratio}$.

---

### 3. La Forma Estructural del Actor (`DirectionalActorForceAligned`)

El actor alinea sus propuestas con las fuerzas físicas subyacentes evaluando la geometría asintótica.

1. **Extracción de Características (Features):** Se calculan los vectores de distancia entre el electrón $i$ y los núcleos ($d_{iI}$) y otros electrones ($d_{ij}$). Las características de entrada son el decaimiento exponencial $e^{-d_{iI}}$ y $e^{-d_{ij}}$.


2. **Vector de Fuerza Direccional:** La red predice pesos escalares $w_I$ y $w_j$ para construir un vector de pseudo-fuerza $\mathbf{f}_{total}$ a partir de los vectores unitarios direccionales $\mathbf{u}_{iI}$ y $\mathbf{u}_{ij}$:



$$\mathbf{f}_{total} = \sum_I w_I \mathbf{u}_{iI} + \sum_j w_j \mathbf{u}_{ij}$$



El drift final emitido aplica una tangente hiperbólica acotada por una amplitud máxima `actor_amplitude` ($A$):



$$\mathbf{A}_\phi(\mathbf{R}) = A \tanh(\mathbf{f}_{total})$$


3. **Matriz de Covarianza (Factorización de Cholesky):** En lugar de predecir una matriz general, la red predice las desviaciones estándar en dirección paralela ($\sigma_\parallel$) y perpendicular ($\sigma_\perp$) al vector unitario $\hat{\mathbf{f}}$.
La matriz de covarianza completa es:



$$\Sigma_\phi = \sigma_\perp^2 \mathbf{I} + (\sigma_\parallel^2 - \sigma_\perp^2) \hat{\mathbf{f}} \hat{\mathbf{f}}^T + \epsilon \mathbf{I}$$



*Truco Numérico:* Las salidas crudas para las sigmas se recortan rígidamente en el intervalo $[-5.0, 1.0]$ antes de aplicar la función exponencial para garantizar que la varianza nunca colapse a 0 ni explote a infinito.



---

### 4. Recompensa (Reward) y Aprendizaje del Actor (RL)

El actor se entrena utilizando el algoritmo REINFORCE (Policy Gradient). La función de recompensa está diseñada para maximizar la exploración física útil penalizando saltos en zonas de alto gradiente.

La recompensa por paso $r_t$ en `GroundStateTrainer` está definida por hiperparámetros $\beta$, $\gamma$ y `penalty_factor`:


$$r_t = \beta (d_{prop} \cdot \alpha_{exact}) - \gamma \Vert{}\nabla \ln \vert{}\Psi\vert{}\Vert{} - \lambda_{pen} (1 - \alpha_{exact})$$

Donde los componentes son:

* **Aceptación Suave ($\alpha_{exact}$):** Para evitar derivadas nulas en la función escalón, la probabilidad de aceptación se aproxima con una forma sigmoidea Barker: $\alpha_{exact} = \exp(\ln \rho - \text{softplus}(\ln \rho))$.


* **Término de Distancia ($d_{prop}$):** Es la distancia euclidiana del salto propuesta, acotada superiormente por `max_phys_jump_std` $\times \sqrt{\tau}$.


* **Penalidad por Gradiente:** Se castiga proporcionalmente a la norma del gradiente del Ansatz, limitando numéricamente este castigo a un máximo de 50.0.



Finalmente, para estabilizar el gradiente de la política, se calcula la Ventaja ($A_t$) normalizando las recompensas con su media y desviación estándar de lote:


$$A_t = \frac{r_t - \mu_r}{\sigma_r + 10^{-8}}$$


La función de pérdida final que se minimiza por descenso de gradiente es:


$$\mathcal{L}_{actor}(\phi) = - \mathbb{E} \left[ A_t \cdot \ln q_\phi(\mathbf{R}' \vert{} \mathbf{R}) \right]$$


*Truco Numérico:* Se utiliza `jax.lax.stop_gradient` sobre la Ventaja $A_t$ para evitar que el optimizador intente derivar a través de la función de recompensa. Además, el Actor cuenta con un "Warm-Up" en el cual su tasa de aprendizaje se mantiene en cero durante pasos iniciales (`actor_warmup_steps`), permitiendo que el muestreador MALA termalice la cadena antes de que la red comience a aprender.