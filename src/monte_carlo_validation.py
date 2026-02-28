import numpy as np
from src.sde_simulation import simulate_controlled_sde_em_with_Z
from src.sde_simulation import simulate_controlled_sde_em


def estimate_risk_neutral_cost(
    *,
    X: np.ndarray,
    U: np.ndarray,
    dt: float,
    alpha: float,
    qT: float,
):
    """
    Estimate risk-neutral cost:
      J = E[ sum_{n=0}^{Nt-1} (X_n^2 + alpha U_n^2) dt + qT X_T^2 ]

    X: (Nt+1, M)
    U: (Nt, M)
    """
    Xn = X[:-1, :]  # (Nt, M)
    XT = X[-1, :]   # (M,)

    running = (Xn**2 + alpha * (U**2)).sum(axis=0) * dt  # per-path
    terminal = qT * (XT**2)
    total = running + terminal

    mean = float(np.mean(total))
    std = float(np.std(total, ddof=1))
    se = std / np.sqrt(total.shape[0])
    return mean, std, se


def run_mc_comparison(
    *,
    M: int,
    x0: float,
    T: float,
    Nt_mc: int,
    sigma: float,
    alpha: float,
    qT: float,
    pde_control_fn,
    riccati_control_fn,
    seed: int = 123,
):
    """
    Compare 3 controllers via Monte Carlo:
      - PDE policy (feedback interpolated from U-grid)
      - Riccati policy (analytic feedback)
      - Zero control baseline
    """
    dt = T / Nt_mc

    # PDE policy
    t_pde, X_pde, U_pde = simulate_controlled_sde_em(
        x0=x0, T=T, Nt=Nt_mc, sigma=sigma,
        control_fn=pde_control_fn, M=M, seed=seed
    )
    J_pde = estimate_risk_neutral_cost(X=X_pde, U=U_pde, dt=dt, alpha=alpha, qT=qT)

    # Riccati policy
    t_ric, X_ric, U_ric = simulate_controlled_sde_em(
        x0=x0, T=T, Nt=Nt_mc, sigma=sigma,
        control_fn=riccati_control_fn, M=M, seed=seed + 1
    )
    J_ric = estimate_risk_neutral_cost(X=X_ric, U=U_ric, dt=dt, alpha=alpha, qT=qT)

    # Zero control
    def zero_control(t, X):
        return np.zeros_like(X)

    t0, X0, U0 = simulate_controlled_sde_em(
        x0=x0, T=T, Nt=Nt_mc, sigma=sigma,
        control_fn=zero_control, M=M, seed=seed + 2
    )
    J0 = estimate_risk_neutral_cost(X=X0, U=U0, dt=dt, alpha=alpha, qT=qT)

    return {
        "pde": {"J_mean": J_pde[0], "J_std": J_pde[1], "J_se": J_pde[2]},
        "riccati": {"J_mean": J_ric[0], "J_std": J_ric[1], "J_se": J_ric[2]},
        "zero": {"J_mean": J0[0], "J_std": J0[1], "J_se": J0[2]},
    }


def run_mc_comparison_crn(
    *,
    M: int,
    x0: float,
    T: float,
    Nt_mc: int,
    sigma: float,
    alpha: float,
    qT: float,
    pde_control_fn,
    riccati_control_fn,
    seed: int = 123,
):
    dt = T / Nt_mc

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((Nt_mc, M))  # common noise

    # PDE
    _, X_pde, U_pde = simulate_controlled_sde_em_with_Z(
        x0=x0, T=T, Nt=Nt_mc, sigma=sigma, control_fn=pde_control_fn, Z=Z
    )
    J_pde = estimate_risk_neutral_cost(X=X_pde, U=U_pde, dt=dt, alpha=alpha, qT=qT)

    # Riccati
    _, X_ric, U_ric = simulate_controlled_sde_em_with_Z(
        x0=x0, T=T, Nt=Nt_mc, sigma=sigma, control_fn=riccati_control_fn, Z=Z
    )
    J_ric = estimate_risk_neutral_cost(X=X_ric, U=U_ric, dt=dt, alpha=alpha, qT=qT)

    # Zero
    def zero_control(t, X):
        return np.zeros_like(X)

    _, X0, U0 = simulate_controlled_sde_em_with_Z(
        x0=x0, T=T, Nt=Nt_mc, sigma=sigma, control_fn=zero_control, Z=Z
    )
    J0 = estimate_risk_neutral_cost(X=X0, U=U0, dt=dt, alpha=alpha, qT=qT)

    # Compute paired difference statistics PDE - Riccati to see if PDE is significantly better than Riccati on this noise realization.
    Xn_pde = X_pde[:-1, :]
    XT_pde = X_pde[-1, :]
    path_cost_pde = (Xn_pde**2 + alpha * (U_pde**2)).sum(axis=0) * dt + qT * (XT_pde**2)

    Xn_ric = X_ric[:-1, :]
    XT_ric = X_ric[-1, :]
    path_cost_ric = (Xn_ric**2 + alpha * (U_ric**2)).sum(axis=0) * dt + qT * (XT_ric**2)

    diff = path_cost_pde - path_cost_ric
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff, ddof=1))
    diff_se = diff_std / np.sqrt(M)

    return {
        "pde": {"J_mean": J_pde[0], "J_std": J_pde[1], "J_se": J_pde[2]},
        "riccati": {"J_mean": J_ric[0], "J_std": J_ric[1], "J_se": J_ric[2]},
        "zero": {"J_mean": J0[0], "J_std": J0[1], "J_se": J0[2]},
        "pde_minus_riccati": {"mean": diff_mean, "std": diff_std, "se": diff_se},
    }

def estimate_risk_sensitive_cost_logexp(
    *,
    X: np.ndarray,
    U: np.ndarray,
    dt: float,
    alpha: float,
    qT: float,
    theta: float,
):
    """
    Risk-sensitive cost:
      J_theta = (1/theta) * log E[ exp( theta * ( sum (X^2 + alpha U^2) dt + qT X_T^2 ) ) ]

    Uses a numerically stable log-sum-exp estimator.

    Returns: (J_hat, se_approx)
    where se_approx is a rough SE from the delta method.
    """
    assert theta > 0.0

    Xn = X[:-1, :]
    XT = X[-1, :]

    path_cost = (Xn**2 + alpha * (U**2)).sum(axis=0) * dt + qT * (XT**2)  # (M,)

    z = theta * path_cost
    zmax = np.max(z)
    # log mean exp
    log_mean_exp = zmax + np.log(np.mean(np.exp(z - zmax)) + 1e-300)
    J_hat = log_mean_exp / theta

    # rough SE (delta method):
    # Var(log mean exp) approx Var(exp(z-zmax))/ (M * mean(exp(z-zmax))^2)
    w = np.exp(z - zmax)
    m = np.mean(w)
    v = np.var(w, ddof=1)
    se_log = np.sqrt(v / (len(w) * (m**2 + 1e-300)))
    se_J = se_log / theta

    return float(J_hat), float(se_J)