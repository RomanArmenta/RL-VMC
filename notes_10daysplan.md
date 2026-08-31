Yes. For a 10-day sprint, I would keep the first paper deliberately narrow. The paper should not try to solve Hartree–Fock, CASSCF, transferability, 3D molecules, alternative RL algorithms, and nodal optimization all at once.

The central paper should be:

$$\boxed{\text{Can TD3 learn a Metropolis-corrected proposal that makes VQMC sampling more efficient?}}$$

for an interacting many-electron 1D harmonic trap, while preserving exact sampling from $\vert{}\Psi\vert{}^2$.

For this first version, I think JCP is the more natural target. The model is deliberately controlled and physics-oriented, and the methodological contribution is about Monte Carlo sampling and many-body quantum calculations. JCTC becomes more attractive once we demonstrate molecules/atoms or a more chemistry-oriented benchmark.

### Proposed Title

Something along the lines of:
**Reinforcement-Learned Metropolis Proposals for Variational Quantum Monte Carlo**
or more explicit:
**Adaptive Variational Quantum Monte Carlo Sampling with Twin Delayed Deep Deterministic Policy Gradients**

The key result should not be “RL gets a better VMC energy.” At fixed wavefunction, it should not.

The claim is instead:

$$\boxed{\text{TD3 reduces correlation and increases effective samples per unit time while preserving VMC observables.}}$$

Then, in coupled wavefunction/sampler optimization, we can investigate whether this improved sampling also makes VMC optimization more effective.

## 10-Day Submission Plan

| Day | Goal | Concrete Deliverable |
| --- | --- | --- |
| **1** | Freeze scientific scope + final computational protocol | Complete experiment matrix, repository frozen, manuscript skeleton |
| **2** | Establish physical and numerical baselines | HO/MALA benchmarks + validation figures |
| **3** | Main TD3-vs-MALA experiments | Core efficiency results for several $N$ and $g$ |
| **4** | Correlation-strength study | Show when RL becomes useful as the system gets harder |
| **5** | Actor/reward ablations | Minimal but convincing explanation of why it works |
| **6** | Production calculations | Definitive frozen-$\Psi$, frozen-actor statistics |
| **7** | Finish all publication figures | Essentially complete Results section |
| **8** | Write Methods + Results | Full technical manuscript draft |
| **9** | Introduction, discussion, references, internal criticism | Full polished manuscript |
| **10** | Final checks + submission | JCP-ready PDF, SI, code/data archive, cover letter |

Let me make each day much more concrete.

## Day 1 — Freeze the paper

This is probably the most important day. Do not keep adding features after today unless something is demonstrably wrong.

I would define the Hamiltonian exactly as:

$$H= -\frac12\sum_{i=1}^N \frac{\partial^2}{\partial x_i^2} + \frac12\omega^2 \sum_i x_i^2 + g\sum_{i<j} \frac{1} {\sqrt{(x_i-x_j)^2+a^2}}.$$

Choose one fixed softening $a$ and one $\omega$, and use $g$ as the correlation knob.
A reasonable primary grid might be:


$$N=2,\;4,\;6,\;8$$


and


$$g=0.5,\;1,\;2,\;4$$


or a smaller matrix if runtime is prohibitive.

I would not make every combination equally expensive. Pick perhaps three representative regimes:


$$g_{\rm weak},\qquad g_{\rm intermediate},\qquad g_{\rm strong}.$$

**Freeze the primary wavefunction**
For the sampler paper, use one reasonably good correlated wavefunction for the main results. I would favor something like:

$$\boxed{\text{Slater + spin-resolved RBF correlation}}$$

rather than the most complicated backflow model. Why? Because we want the sampling problem to be nontrivial, but we don't want reviewers arguing that all gains come from a pathological wavefunction architecture. Backflow can appear in the SI or one robustness experiment.

**Freeze the main sampler comparison**
Primary:


$$\text{MALA} \quad\text{vs}\quad \text{TD3 residual-MALA}.$$


Do not make SAC/PPO part of this first article.

**Freeze the actor**
Use the new:


$$\boxed{\text{multi-branch DeepSet actor}}$$


with trainable radial embedding as the primary model. Keep the flat MLP only as an ablation.

**Paper skeleton by the end of Day 1**
Create:

* Introduction
* Model and VQMC
* Reinforcement-learned proposal
* Computational protocol
* Results
* Discussion
* Conclusions

And already put empty figure slots into the manuscript.

## Day 2 — Prove that the physics and sampler are correct

Before showing RL, establish credibility.

**Noninteracting benchmark**
At $g=0$, the exact energy is:

$$E_0 = \omega \left[ \sum_{n=0}^{N_\uparrow-1}\left(n+\frac12\right) + \sum_{n=0}^{N_\downarrow-1}\left(n+\frac12\right) \right].$$

Show the VMC code reproduces it. This might go to SI, but it must exist.

**MALA correctness**
For selected systems, demonstrate:


$$\langle E\rangle_{\rm MALA} = \langle E\rangle_{\rm RL}$$


within statistical uncertainty. Use:


$$z_E= \frac{ E_{\rm RL}-E_{\rm MALA} }{ \sqrt{ \mathrm{SE}_{\rm RL}^2+ \mathrm{SE}_{\rm MALA}^2 } }.$$


We want:


$$\vert{}z_E\vert{}\lesssim2.$$


Also compare at least one distribution:


$$\rho(x)$$


or


$$P(r_{ij}).$$


This directly demonstrates that the learned proposal has not changed the target distribution.

**Figure 1**
I would make Figure 1 the methodology figure:


$$\Psi \rightarrow F_i \rightarrow \text{actor} \rightarrow (\Delta\mu_i,\sigma_i) \rightarrow \text{proposal} \rightarrow \text{MH correction}.$$


This is essentially the actor graphic we've already developed, but paper-style.

## Day 3 — The central result: MALA vs TD3

Run the definitive frozen-wavefunction experiments.

For every test:


$$\boxed{\Psi\;\text{fixed}}$$


and


$$\boxed{\pi_{\rm TD3}\;\text{fixed during production}}$$


after training.

Measure:


$$\text{acceptance},\quad \langle \vert{}\Delta x\vert{}\rangle,\quad \mathrm{ESJD} = \left\langle(\Delta x)^2\right\rangle,\quad \tau_{\rm int}(E_L),\quad N_{\rm eff},\quad \frac{N_{\rm eff}}{t},\quad \mathrm{SE}(E).$$

The quantity I would emphasize most heavily is:

$$\boxed{ G_{\rm RL} = \frac{ (\mathrm{ESS}/t)_{\rm RL} }{ (\mathrm{ESS}/t)_{\rm MALA} }. }$$

This is the cleanest sampler-efficiency number.

**Figure 2**
For one representative challenging system:

* **A:** local-energy time series
* **B:** autocorrelation function
* **C:** energy histogram
* **D:** running energy uncertainty (MALA vs TD3).

This should visually establish the story.

## Day 4 — Make the result physically interesting

Now sweep interaction strength. Define approximately:

$$\lambda = \frac{g}{\sqrt{\omega}}$$

as an interaction/confinement measure, with the caveat that the effective coupling also depends on $a\sqrt{\omega}$.

Then plot:


$$G_{\rm RL}(\lambda).$$

The potential story is very compelling if we see:


$$G_{\rm RL}\approx1 \qquad\text{weak correlation},$$


but


$$G_{\rm RL}>1$$


and increasing as correlation gets stronger.

Then the result becomes more than: *neural networks make MCMC faster.*
It becomes: *the benefit of learned proposals grows as the many-body distribution becomes more difficult.*

**Figure 3**
Something like $\mathrm{ESS/s}$ versus $g$ for MALA and TD3. And/or $G_{\rm RL}$ versus $g$, with separate curves for $N$. This could become the headline result.

## Day 5 — Explain what the actor learned

This is where we make the ML part scientifically satisfying. We do not need twenty ablations. I would do exactly three.

**A. Scale-only vs residual-MALA**
Compare:


$$\mu_i= x_i+\frac12\sigma_i^2F_i$$


against


$$\mu_i= x_i+ \frac12\sigma_i^2F_i+ \Delta\mu_i.$$


This tells us whether the improvement is simply adaptive step size or genuinely learned directional corrections.

**B. Flat MLP vs structured actor**
Compare `flat_mlp` against `multibranch_deepset`. This justifies the architecture we developed.

**C. Reward**
Keep the current successful reward:


$$r_{\rm old} = 2\beta\log(\vert{}\Delta x\vert{}+\epsilon) + \gamma_\alpha\log(\alpha+\epsilon),$$


and compare against the proposed expected-squared-jump-distance-inspired reward:


$$r_{\rm ESJD} = \log \left[ \epsilon+ \alpha \left( \frac{\Delta x}{\epsilon_0} \right)^2 \right].$$


Do not necessarily optimize these extensively. A representative difficult system is enough.

**Figure 4**
A compact ablation plot of $\mathrm{ESS/s}$ for:

* MALA
* scale-only TD3
* residual-MALA flat actor
* residual-MALA DeepSet actor

Possibly put reward comparison in SI if the figure becomes crowded.

## Day 6 — Definitive production calculations

This is where the new production framework becomes essential. For the key systems, run long chains with:


$$\boxed{\text{frozen }\Psi}$$


and


$$\boxed{\text{frozen actor}}.$$

Use independent burn-in: $B$ sweeps discarded. Do not thin.

Then use the autocorrelation-corrected error:


$$\mathrm{SE}(E) = \sqrt{ \frac{\mathrm{Var}(E_L)} {N_{\rm eff}} }.$$

This day should answer the practically important question: *How many samples are needed to obtain a given uncertainty?*

Choose a target, e.g. $\mathrm{SE}(E)<10^{-3}\;E_h$ or whatever is appropriate for the energy scale.
Define $T_\epsilon = \text{time required to reach } \mathrm{SE}(E)<\epsilon.$
Then compare:

$$\boxed{ \frac{T_\epsilon^{\rm MALA}} {T_\epsilon^{\rm RL}} }.$$

This may be even more intuitive than ESS/sec.

**Figure 5**
Plot $\mathrm{SE}(E)$ against either number of sweeps and/or wall time. The wall-time plot is particularly powerful.

## Day 7 — Finish figures before finishing prose

By the end of Day 7, I would require every principal scientific claim to have a finished figure. My preferred five-figure JCP paper would be:

* **Figure 1 — Method:** Hamiltonian + VQMC + residual-MALA + TD3 actor + exact Metropolis-Hastings correction.
* **Figure 2 — Representative chain:** MALA vs TD3: $E_L(t),\quad \rho_E(k),\quad P(E_L),\quad \text{running error}.$
* **Figure 3 — Main scaling result:** $\mathrm{ESS/s}$ or relative gain versus $N,\;g.$
* **Figure 4 — What did RL learn?:** Proposal statistics: $P(\sigma), \qquad P(\Delta\mu), \qquad \vert{}\Delta x\vert{},$ plus perhaps learned radial-embedding centers.
* **Figure 5 — Ablations / time to precision:** Either architectural ablations or $\mathrm{SE}(E)\;\text{vs wall time}.$ If necessary, move the ablations to SI and let Figure 5 be the stronger time-to-solution result.

## Day 8 — Write Methods and Results

Start with the technical sections, not the introduction.

**Methods needs five pieces**

1. **Hamiltonian:** $H=T+V_{\rm trap}+V_{ee}.$
2. **Wavefunction:** Explain determinant × correlation factor, but don't let wavefunction architecture dominate the manuscript.
3. **Baseline MALA:** $x_i' = x_i+ \frac{\epsilon^2}{2} \partial_{x_i}\log\vert{}\Psi\vert{}^2 + \epsilon\eta.$
4. **Learned proposal:** $x_i' \sim \mathcal N(\mu_i,\sigma_i^2),$ with $\mu_i= x_i+ \frac12\sigma_i^2F_i+ \Delta\mu_\theta(s_i).$
5. **Exact Metropolis correction:**

$$\alpha= \min \left[ 1, \frac{ \vert{}\Psi(X')\vert{}^2q(X\vert{}X') }{ \vert{}\Psi(X)\vert{}^2q(X'\vert{}X) } \right].$$



This equation should be prominent. It immediately answers a likely reviewer concern: *Does RL bias the VMC distribution?* No, because RL parameterizes the proposal; Metropolis-Hastings still defines the invariant distribution.

Then describe TD3: shared actor, twin critics, replay buffer, delayed actor update, target smoothing. Don't spend three pages teaching TD3.

**Results**
Write results in exactly the order of the figures.

## Day 9 — Write the scientific story

Now write the Introduction and Discussion.

I would structure the Introduction around this problem:
VQMC itself has two optimization problems:


$$\boxed{\text{represent the wavefunction}}$$


and


$$\boxed{\text{efficiently sample }\vert{}\Psi\vert{}^2.}$$


Most ML-VMC work emphasizes the first. Your paper tackles the second.

Then:
MALA incorporates local wavefunction geometry through $\nabla\log\vert{}\Psi\vert{}^2,$ but its proposal functional form remains prescribed. We ask whether RL can learn state-dependent deviations from MALA while Metropolis-Hastings guarantees correctness. That is the conceptual novelty.

**Discussion should be careful**
Explicitly say that at fixed $\Psi$:


$$E_{\rm VMC}^{\rm RL} = E_{\rm VMC}^{\rm MALA}$$


within sampling uncertainty. RL does not improve the variational ansatz.
It improves:


$$\tau_{\rm int}, \qquad N_{\rm eff}, \qquad N_{\rm eff}/t, \qquad \mathrm{SE}(E;t).$$


That distinction will make the paper much stronger.

## Day 10 — Reviewer simulation and submission

I would spend the first half of Day 10 trying to reject our own paper.

**Ask:**

* **Is RL actually sampling the same distribution?** Show observable agreement + exact MH derivation.
* **Is TD3 just learning a larger step size?** Scale-only ablation.
* **Does larger displacement actually improve sampling?** Show autocorrelation and ESS/sec.
* **Are improvements just caused by a bad MALA step-size choice?** Very important: optimize the MALA step size reasonably for every system. This is essential. We should compare RL against a tuned MALA, not an intentionally mediocre fixed MALA. Perhaps perform $\epsilon\in \{0.05,0.10,0.15,\ldots\}$ and use the MALA step giving the best $\mathrm{ESS/s}.$ Otherwise a reviewer will immediately say the RL actor merely learned step-size tuning.
* **Does the NN cost more than the saved sampling?** Wall-clock ESS/sec answers this.
* **Why 1D?** Because it provides a controlled many-electron testbed where correlation strength, particle number, nodal structure, and reference calculations are systematically accessible. Do not apologize for it. Present it as controlled methodological validation.

**Then package:**
Manuscript PDF, Supporting Information, source/data availability statement, Git repository release, seeds/configuration files, production CSVs, cover letter.

## What I would keep out of Paper 1

This is important for hitting ten days. I would not wait for:

$$\text{Hartree–Fock orbital initialization},\quad \text{CASSCF},\quad \text{SAC},\quad \text{PPO},\quad \text{3D atoms},\quad \text{molecular systems},\quad \text{transfer learning},$$

or joint-electron covariance policies. Those are Paper 2 / follow-up material. The first paper needs one sharp result.

## The scientific story I would aim for

The abstract-level story would eventually be something like:

> We formulate adaptive VQMC sampling as a reinforcement-learning problem in which a Twin Delayed Deep Deterministic Policy Gradient agent learns state-dependent residual corrections and proposal scales on top of a Metropolis-adjusted Langevin proposal. Exact Metropolis-Hastings correction preserves the target distribution. For interacting electrons in one-dimensional harmonic confinement, learned proposals produce larger effective moves, lower autocorrelation times, and increased effective sample rates relative to optimized MALA, with the advantage increasing in strongly correlated regimes.

That is enough for a paper if the numerical evidence is clean. And the central quantitative panel should ideally let us make a sentence of the form:

$$\boxed{\text{“TD3 increases ESS/s by }X{-}Y\times \text{ and reduces time-to-precision by }Z\%.”}$$

I would build the entire 10-day sprint around obtaining those two numbers reliably.