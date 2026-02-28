# Risk-Sensitive Stochastic Control via Nonlinear HJB (Policy Iteration + Monte Carlo)

This repository studies a **risk-sensitive linear-quadratic stochastic control** problem and implements:

- Formulation of the **nonlinear risk-sensitive HJB**
- A **finite-difference PDE solver** (implicit diffusion + upwind drift)
- **Policy iteration** (Howard / fixed-point) for the optimal feedback control
- **Risk-neutral vs risk-averse** comparisons
- **Monte Carlo validation** (Euler-Maruyama) of both mean and exponential (risk-sensitive) criteria
- **Sensitivity analysis** in the risk parameter $\theta$ and volatility $\sigma$

---

## 1) Problem setup

### Controlled diffusion
Consider the 1D controlled SDE

$$
dX_t = u_t\,dt + \sigma\, dW_t, \qquad t\in[0,T],
$$

where $u_t$ is the control and $\sigma>0$ is the volatility.

### Running and terminal costs
Running cost:

$$
\ell(x,u) = x^2 + \alpha u^2,\qquad \alpha>0.
$$

Terminal cost (used in the experiments):

$$
g(x)= q_T x^2.
$$

### Risk-sensitive objective
For $\theta\neq 0$ (risk-sensitive / exponential-of-integral criterion),

$$
J_\theta(u) = \frac{1}{\theta}\log \mathbb{E}\left[\exp\left(\theta\Big(\int_0^T \ell(X_t,u_t)\,dt + g(X_T)\Big)\right)\right].
$$

Risk-neutral is recovered as $\theta\to 0$:

$$
J_0(u) = \mathbb{E} \left[\int_0^T \ell(X_t,u_t)\,dt + g(X_T)\right].
$$

---

## 2) Risk-sensitive HJB derivation (core equation)

Let the value function be

$$
V(t,x)=\inf_u J_\theta^{t,x}(u),
$$

where the superscript indicates starting from $X_t=x$.

For the risk-sensitive criterion, the dynamic programming equation yields the nonlinear HJB:

$$
V_t + \inf_{u}
\bigl( x^2 + \alpha u^2 + u V_x + \tfrac{\sigma^2}{2} V_{xx} + \tfrac{\theta \sigma^2}{2} (V_x)^2 \bigr) = 0,
\qquad V(T,x) = g(x)
$$

The minimizer of the Hamiltonian is explicit:

$$
\frac{\partial}{\partial u}\Big(\alpha u^2 + uV_x\Big)=2\alpha u + V_x=0
\quad\Rightarrow\quad
u^*(t,x)= -\frac{V_x(t,x)}{2\alpha}.
$$

Substituting $u^*$ back gives the reduced nonlinear PDE:

$$
V_t+ x^2+\frac{\sigma^2}{2}V_{xx}+\left(\frac{\theta\sigma^2}{2}-\frac{1}{4\alpha}\right)(V_x)^2=0.
$$

> **Nonlinearity:** the $(V_x)^2$ term (present when $\theta\neq 0$)

---

## 3) Numerical method

On a bounded spatial domain

$$
x\in[-X_{\max},X_{\max}],\qquad t\in[0,T],
$$

is solved using a uniform grid with $(N_t,N_x)$ points.

### Boundary and terminal conditions
- Terminal: $V(T,x)=q_T x^2$
- Dirichlet boundaries (stable and simple):

$$
V(t,\pm X_{\max})=\kappa X_{\max}^2
$$

### Discretization choices
The evaluation PDE ( for a fixed control field $u(t,x)$ ) is:

$$
V_t + \ell(x,u) + uV_x + \frac{\sigma^2}{2}V_{xx} + \frac{\theta\sigma^2}{2}(V_x)^2 = 0.
$$

Implemented scheme:
- **Implicit** in diffusion ($V_{xx}$) via a tri-diagonal solve each timestep
- **Explicit** in drift ($uV_x$) and running cost $\ell$
- **Upwind** finite differences for $V_x$ in the drift term (stability for advection)

This yields, backward in time, a linear tri-diagonal system at each $t_n$.

---

## 4) Policy iteration (Howard-style)

I iteratively alternate:

1. **Policy evaluation**: given $U^{(k)}(t,x)$ solve for $V^{(k)}(t,x)$
2. **Policy improvement**:

$$
U^{(k+1)}(t,x) = \Pi_{[-u_{\max},u_{\max}]}\left(-\frac{V_x^{(k)}(t,x)}{2\alpha}\right),
$$

where $\Pi$ is clipping to stabilize numerics.

### Handling the risk-sensitive nonlinearity
For $\theta>0$, evaluation uses a **Picard** (fixed-point) loop for the $(V_x)^2$ term:
- Freeze $(V_x)^2$ using a previous iterate to keep each inner solve linear
- Apply a relaxation step to stabilize convergence

---

## 5) Monte Carlo validation (Euler–Maruyama)

Validate computed feedback policies $u^*(t,x)$ by simulating:

$$
X_{k+1} = X_k + u(t_k,X_k)\,\Delta t + \sigma\sqrt{\Delta t}\,Z_k,\qquad Z_k\sim\mathcal{N}(0,1).
$$

Estimators:
- **Risk-neutral mean cost**

$$
\widehat{J}_0 = \frac{1}{M}\sum_{i=1}^M\left(\sum_{k}\ell(X_k^{(i)},u_k^{(i)})\Delta t + g(X_T^{(i)})\right).
$$

- **Risk-sensitive exponential cost** (evaluated at some $\theta_{\text{eval}}$)

$$
\widehat{J}_{\theta_{\text{eval}}} = \frac{1}{\theta_{\text{eval}}}\log\left( \frac{1}{M}\sum_{i=1}^M \exp\!\left(\theta_{\text{eval}}\Big(\sum_k \ell(X_k^{(i)},u_k^{(i)})\Delta t + g(X_T^{(i)})\Big)\right) \right).
$$

---

## 6) Repository layout

Key modules:

- `src/utils.py`  
  Grid builder + finite-difference operators (`make_grid`, upwind derivative).
- `src/linalg.py`  
  Tri-diagonal solver (`solve_tridiagonal`).
- `src/hjb_solver.py`  
  Main HJB + policy iteration solver.
- `src/riccati_benchmark.py`  
  Risk-neutral LQR benchmark via Riccati (for sanity checking).
- `src/sde_simulation.py`  
  Euler–Maruyama simulation under a feedback control.
- `src/monte_carlo_validation.py`  
  MC cost estimators + error bars.

Experiments (invoked as modules):
- `python -m experiments.risk_neutral_case`
- `python -m experiments.risk_averse_case`
- `python -m experiments.monte_carlo_risk_neutral`
- `python -m experiments.monte_carlo_risk_sensitive`
- `python -m experiments.parameter_sensitivity` 
- `python -m experiments.sigma_sensitivity`

Outputs are saved under:
- `results/plots/` (PNG figures)
- `results/tables` (CSV summaries)

---

## 7) How to run

### Install
```bash
pip install numpy scipy matplotlib pandas 
```

### Run main PDE solves
```bash
python -m experiments.risk_neutral_case
python -m experiments.risk_averse_case
```

### Run Monte Carlo validation
```bash
python -m experiments.monte_carlo_risk_neutral
python -m experiments.monte_carlo_risk_sensitive
```

### Run sensitivity sweeps
```bash
python -m experiments.parameter_sensitivity
python -m experiments.sigma_sensitivity
```

---

## 8) Final grid + MC settings used 

### PDE grid (refined)
Risk-neutral and risk-averse experiments used:
- $T=1.0$
- $X_{\max}=10.0$
- $N_t=2000$
- $N_x=1201$
- $\sigma=0.7$, $\alpha=1.0$
- Control clip $u_{\max}=8.0$
- Dirichlet scale $\kappa=1.0$
- Terminal scale $q_T=1.0$

### Monte Carlo (refined)
- $M=50000$ paths  
- $x_0=1.0$
- $N_{t,\text{MC}}=2000$
- Fixed RNG seed for reproducibility

---

## 9) Key results

### 9.1 Risk-neutral: PDE vs Riccati benchmark (high-accuracy check)

With the refined grid, the PDE feedback policy matches the Riccati solution very closely:

- **Masked $L^2$ error:** $6.03\times 10^{-3}$
- **Masked $L^\infty$ error:** $6.32\times 10^{-3}$
- **Relative $L^2$ error:** $1.39\times 10^{-3}$

Monte Carlo mean-cost comparison (same simulated noise, paired):
- PDE policy: $J=1.496477$ (SE $0.004716$)
- Riccati policy: $J=1.496451$ (SE $0.004709$)
- Zero control: $J=2.750168$ (SE $0.010409$)
- Paired diff (PDE − Riccati): mean $2.59\times 10^{-5}$ (SE $9.24\times 10^{-6}$)

**Interpretation:** the PDE solver + policy iteration recovers the classical LQR solution in the $\theta=0$ limit.

### 9.2 Risk aversion changes the controller (theta = 1)

At $t=0$:
- risk-neutral $U(0,x)$ is essentially linear near $x=0$
- risk-averse control has **larger feedback gain** near the origin (more negative $u/x$), i.e. stronger pull toward $0$

### 9.3 Risk-sensitive validation (theta_eval = 1.0)

Monte Carlo estimates (same $x_0=1$, $M=20000$):
- **Mean cost** (risk-neutral objective):
  - $\theta=0$ policy: $1.496477$ (SE $0.004716$)
  - $\theta=1$ policy: $1.541126$ (SE $0.004355$)

- **Exponential cost** (risk-sensitive objective, $\theta_{\text{eval}}=1$):
  - $\theta=0$ policy: $2.848614$ (SE $\approx 0.099927$)
  - $\theta=1$ policy: $2.588342$ (SE $\approx 0.067445$)

**Interpretation:** the risk-averse policy can sacrifice mean performance while improving the exponential criterion.

---

## 10) Sigma-theta sensitivity: what the plots show

The sensitivity sweep computes:
- feedback-gain proxy near $x=0$:

$$
\text{gain}(\theta,\sigma) \approx \mathbb{E}_{|x|\le 1}\left[\frac{u^*(0,x)}{x}\right],
$$

- Monte Carlo mean cost $\widehat{J}_0$
- Monte Carlo exponential cost $\widehat{J}_{\theta}$ (evaluated at the same $\theta$ in the sweep)

### Main qualitative takeaways

1. **Gain magnitude increases with $\theta$ (more negative $u/x$)**  
   For each fixed $\sigma$, increasing $\theta$ makes the controller more aggressive near the origin.

2. **Risk aversion interacts strongly with volatility**  
   Larger $\sigma$ amplifies the impact of risk sensitivity: the gain curve steepens and the exponential cost grows quickly.

3. **Mean cost grows mildly with $\theta$**  
   The mean-cost curves are relatively flat compared to the exponential-cost curves, consistent with the idea that risk sensitivity is primarily changing tail / variance behavior.

4. **Exponential cost can have a “sweet spot” at moderate $\theta$**

   In the sweep outputs, $\widehat{J}_{\theta}$ decreases from $\theta=0$ to a small positive $\theta$ for several $\sigma$ values, then increases for larger $\theta$.

   This can happen because moderate risk sensitivity reduces variability, while too-large $\theta$ drives stronger controls and increases the integral cost, and clipping ($u_{\max}$) can also influence the optimum.

---

## 11) Figures produced

The experiment scripts save key figures to `results/plots/` including:

- `U0_vs_riccati.png` -- risk-neutral $u^*(0,x)$ vs Riccati benchmark
- `V0_risk_neutral.png` -- value slice $V(0,x)$ (risk-neutral)
- `U0_risk_neutral.png` -- policy slice $u^*(0,x)$ (risk-neutral)
- `U0_over_x_risk_neutral.png` -- sanity check: $u^*(0,x)/x$ near $0$
- `U0_theta_compare.png` -- $u^*(0,x)$: $\theta=0$ vs $\theta=1$
- `U0_over_x_theta_compare.png` -- gain proxy comparison
- `V0_theta_compare.png` -- $V(0,x)$: $\theta=0$ vs $\theta=1$
- `mc_cost_bar.png` -- MC mean-cost bars (zero vs Riccati vs PDE)
- `gain_vs_theta.png`, `gain_vs_theta_by_sigma.png`
- `Jexp_vs_theta.png`, `Jexp_vs_theta_by_sigma.png`
- `Jmean_vs_theta_by_sigma.png`

---

## 12) Notes and future improvements

- **Risk-sensitive policy iteration is harder** than the risk-neutral case; convergence can be slow or unstable for large $\theta$ and/or large $\sigma$.
- Improvements that can help:
  - stronger damping / line-search in policy iteration
  - smaller $\Delta t$ for the Picard inner loop
  - alternative boundary conditions (e.g. Neumann) and larger $X_{\max}$
  - higher-order / monotone schemes for the nonlinear term
  - acceleration via `numba` or vectorization (especially in Monte Carlo)

---

## References

[1] Jacobson, D. H. (1973). Optimal stochastic linear systems with exponential performance criteria. IEEE Transactions on Automatic Control.

[2] Fleming, W. H., & Soner, H. M. (2006). Controlled Markov Processes and Viscosity Solutions. Springer.

[3] Whittle, P. (1990). Risk-Sensitive Optimal Control. Wiley.

[4] Kushner, H. J., & Dupuis, P. G. (2001). Numerical Methods for Stochastic Control Problems in Continuous Time. Springer.

---